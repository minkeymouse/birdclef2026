#!/usr/bin/env python3
"""exp140 — Download iNat clips for saturating Aves missing external.

These are the Aves species that v3 pseudo over-fired (>5% of unlabeled rows)
but had no external clips for verification. Download up to 50 each.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
EXT_OUT = ROOT / "data/external"
EXT_OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = EXT_OUT / "_logs"; LOG_DIR.mkdir(exist_ok=True)

MAX_PER_SPECIES = 50
RATE_LIMIT_SEC = 1.0
MAX_FILE_SIZE_MB = 10
TIMEOUT_QUERY = 30
TIMEOUT_DL = 60
ALLOWED_EXT = {"mp3", "wav", "ogg", "m4a"}

# Saturating Aves from v3 audit + a few common-Aves candidates
TARGETS = [
    "grepot1", "shtnig1", "compau", "strowl1", "sobcac1", "greani1",
    "whiwoo1", "baffal1", "linwoo1", "toctou1", "picpig2", "grfdov1",
    "smbani", "greyel", "blttit1", "rubthr1", "epaori4",
]


def safe_download(url, out_path):
    if not url.startswith("https://"): return False, "non-https URL"
    try:
        r = requests.get(url, timeout=TIMEOUT_DL, stream=True,
                          headers={"User-Agent": "BirdCLEF2026-research"})
        if r.status_code != 200: return False, f"HTTP {r.status_code}"
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"too large"
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    total += len(chunk)
                    if total > MAX_FILE_SIZE_MB * 1024 * 1024:
                        f.close(); out_path.unlink(missing_ok=True)
                        return False, "size cap exceeded"
                    f.write(chunk)
        return True, total
    except Exception as e:
        return False, str(e)


def query_inat(taxon_id, limit=100):
    url = (f"https://api.inaturalist.org/v1/observations"
           f"?taxon_id={taxon_id}&sounds=true&quality_grade=research&per_page={limit}")
    r = requests.get(url, timeout=TIMEOUT_QUERY,
                      headers={"User-Agent": "BirdCLEF2026-research"})
    if r.status_code != 200: return []
    return r.json().get("results", [])


def convert_to_ogg(src):
    if src.suffix.lower() not in (".m4a", ".mp3"): return
    ogg = src.with_suffix(".ogg")
    if ogg.exists(): return
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                         "-ac", "1", "-ar", "32000", str(ogg)],
                        check=True, timeout=30)
    except Exception:
        pass


def main():
    print("=== exp140: Download more external for saturating Aves ===\n", flush=True)
    tax = pd.read_csv(DATA / "taxonomy.csv")

    summary = {"per_species": {}}
    total_new = 0
    for sp_label in TARGETS:
        row = tax[tax.primary_label.astype(str) == sp_label]
        if len(row) == 0:
            print(f"  [{sp_label}] NOT IN TAXONOMY, skip")
            continue
        r = row.iloc[0]
        taxon_id = int(r.inat_taxon_id)
        sn = r.scientific_name

        sp_dir = EXT_OUT / sp_label
        sp_dir.mkdir(exist_ok=True)
        existing = {p.stem.split(".")[0] for p in sp_dir.glob("inat_*.*")}
        if len(existing) >= MAX_PER_SPECIES:
            print(f"  [{sp_label}] {sn}: already have {len(existing)}, skip")
            continue

        print(f"  [{sp_label}] {sn} (taxon_id={taxon_id}): querying iNat...", flush=True)
        obs = query_inat(taxon_id, limit=MAX_PER_SPECIES * 2)
        print(f"    Got {len(obs)} observations")

        n_new = 0; n_failed = 0
        for o in obs:
            if n_new + len(existing) >= MAX_PER_SPECIES: break
            for s in o.get("sounds", []):
                url = s.get("file_url")
                if not url: continue
                ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
                if ext not in ALLOWED_EXT: continue
                stem = f"inat_{o['id']}_{s['id']}"
                if stem in existing: continue
                out_path = sp_dir / f"{stem}.{ext}"
                ok, _ = safe_download(url, out_path)
                time.sleep(RATE_LIMIT_SEC)
                if ok:
                    n_new += 1
                    existing.add(stem)
                    convert_to_ogg(out_path)
                else:
                    n_failed += 1
                if n_new + len(existing) - len([s2 for s2 in existing if s2 == stem]) >= MAX_PER_SPECIES:
                    break
        print(f"    [{sp_label}] new {n_new}, total now {len(existing)}, failed {n_failed}", flush=True)
        total_new += n_new
        summary["per_species"][sp_label] = {"new": n_new, "total": len(existing), "scientific_name": sn}

    print(f"\nTotal new clips: {total_new}")
    log_path = LOG_DIR / f"exp140_download_{int(time.time())}.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
