#!/usr/bin/env python3
"""exp84 Q4 — Expanded external download for non-Aves species + low-n Aves.

Builds on exp71 (which downloaded 5 targets). Now expands to the full set
of non-Aves species in our taxonomy that have iNat coverage, plus the
weakest-AUC Aves species per audit.

Then re-runs the exp73 fine-tune pipeline with the larger external pool.

Plan:
  Phase 1 (this script): download up to 30 clips per species across 30+
    target species. Skip Insecta sonotypes (47158sonXX have no scientific
    name, can't query). Cap total bandwidth.
  Phase 2 (exp84b script): retrain exp50 base with combined original +
    larger external set, eval on labeled SS.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import pandas as pd
import requests

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
OUT = ROOT / "data" / "external"
OUT.mkdir(exist_ok=True, parents=True)
LOG_DIR = OUT / "_logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_INAT_PER_SPECIES = 30
MAX_FILE_SIZE_MB = 10
RATE_LIMIT_SEC = 1.0
ALLOWED_EXT = {"mp3", "wav", "ogg", "m4a"}
TIMEOUT_QUERY = 30
TIMEOUT_DL = 60


def convert_to_ogg(src_path: Path):
    """If src is m4a/mp3, convert to ogg alongside (so exp84b can load)."""
    if src_path.suffix.lower() not in (".m4a", ".mp3"):
        return
    ogg_path = src_path.with_suffix(".ogg")
    if ogg_path.exists():
        return
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_path),
                         "-ac", "1", "-ar", "32000", str(ogg_path)],
                        check=True, timeout=30)
    except Exception as e:
        print(f"    ffmpeg failed for {src_path.name}: {e}")


def safe_download(url, out_path, max_size_mb=MAX_FILE_SIZE_MB):
    if not url.startswith("https://"): return False, "non-https URL"
    try:
        r = requests.get(url, timeout=TIMEOUT_DL, stream=True,
                          headers={"User-Agent": "BirdCLEF2026-research"})
        if r.status_code != 200: return False, f"HTTP {r.status_code}"
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > max_size_mb * 1024 * 1024:
            return False, f"too large ({int(cl)/1e6:.1f}MB)"
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    total += len(chunk)
                    if total > max_size_mb * 1024 * 1024:
                        f.close(); out_path.unlink(missing_ok=True)
                        return False, "size cap exceeded mid-stream"
                    f.write(chunk)
        return True, total
    except Exception as e:
        return False, str(e)


def query_inat_taxon(taxon_id, limit=30):
    url = (f"https://api.inaturalist.org/v1/observations"
           f"?taxon_id={taxon_id}&sounds=true&quality_grade=research&per_page={limit}")
    r = requests.get(url, timeout=TIMEOUT_QUERY,
                      headers={"User-Agent": "BirdCLEF2026-research"})
    if r.status_code != 200: return []
    return r.json().get("results", [])


def download_inat_for_species(species_label, taxon_id, max_clips=MAX_INAT_PER_SPECIES,
                                already_downloaded: set | None = None):
    """Returns (n_new, n_skipped_existing, n_failed)."""
    sp_dir = OUT / str(species_label)
    sp_dir.mkdir(exist_ok=True)
    if already_downloaded is None:
        already_downloaded = {p.stem for p in sp_dir.glob("inat_*.*")}

    obs = query_inat_taxon(taxon_id, limit=max_clips * 2)
    print(f"  [{species_label}] inat_taxon_id={taxon_id}: {len(obs)} observations queried", flush=True)

    n_new = 0; n_skipped = 0; n_failed = 0
    for o in obs[:max_clips * 2]:
        if n_new >= max_clips: break
        for s in o.get("sounds", []):
            url = s.get("file_url")
            if not url: continue
            ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED_EXT: continue
            stem = f"inat_{o['id']}_{s['id']}"
            if stem in already_downloaded:
                n_skipped += 1; continue
            out_path = sp_dir / f"{stem}.{ext}"
            ok, _ = safe_download(url, out_path)
            time.sleep(RATE_LIMIT_SEC)
            if ok:
                n_new += 1
                already_downloaded.add(stem)
                convert_to_ogg(out_path)
            else:
                n_failed += 1
    return n_new, n_skipped, n_failed


def main():
    print("=== exp84 Q4: expanded external download ===\n", flush=True)
    tax = pd.read_csv(DATA / "taxonomy.csv")

    # Target list:
    # 1. ALL non-Aves species with iNat taxon ID (=72)
    # 2. Plus the weak Aves with low n_train_audio (from audits)
    weak_aves = ["bafcur1", "purjay1", "plcjay1", "rutjac1", "thlwre1", "trsowl",
                  "litnig1", "compot1", "sptnig1", "phecuc1", "ragmac1"]

    targets = []
    # Non-Aves
    for _, r in tax.iterrows():
        if r.class_name in ("Amphibia", "Mammalia", "Reptilia"):
            targets.append((str(r.primary_label), int(r.inat_taxon_id), r.class_name, r.scientific_name))
        elif r.class_name == "Insecta" and not str(r.primary_label).startswith("47158son"):
            # 3 named Insecta: 1161364 (Guyalna cuta), etc.
            targets.append((str(r.primary_label), int(r.inat_taxon_id), r.class_name, r.scientific_name))
    # Weak Aves
    for lbl in weak_aves:
        row = tax[tax.primary_label.astype(str) == lbl]
        if len(row) == 0: continue
        r = row.iloc[0]
        targets.append((lbl, int(r.inat_taxon_id), r.class_name, r.scientific_name))

    print(f"Target species: {len(targets)}", flush=True)
    print(f"  by taxon: {pd.Series([t[2] for t in targets]).value_counts().to_dict()}", flush=True)

    # Download
    summary = {"started": time.time(), "per_species": {}}
    total_new = 0; total_skipped = 0; total_failed = 0
    for sp, tid, cn, sn in targets:
        # Skip if already have full quota
        sp_dir = OUT / str(sp)
        existing = list(sp_dir.glob("inat_*.*")) if sp_dir.exists() else []
        existing_stems = {p.stem.split(".")[0] for p in existing}
        if len(existing_stems) >= MAX_INAT_PER_SPECIES:
            print(f"  [{sp}] {sn} ({cn}): already have {len(existing_stems)}, skip", flush=True)
            summary["per_species"][sp] = {"new": 0, "skipped": len(existing_stems), "failed": 0,
                                            "scientific_name": sn, "class": cn}
            continue
        try:
            new, skip, fail = download_inat_for_species(sp, tid,
                                                         max_clips=MAX_INAT_PER_SPECIES - len(existing_stems),
                                                         already_downloaded=existing_stems)
        except Exception as e:
            print(f"  [{sp}] ERROR: {e}", flush=True)
            new, skip, fail = 0, 0, 0
        summary["per_species"][sp] = {"new": new, "skipped": skip, "failed": fail,
                                        "scientific_name": sn, "class": cn}
        total_new += new; total_skipped += skip; total_failed += fail
        print(f"  [{sp}] {sn} ({cn}): +{new} new, {skip} existed, {fail} failed", flush=True)

    summary["totals"] = {"new": total_new, "skipped": total_skipped, "failed": total_failed,
                          "elapsed_sec": time.time() - summary["started"]}

    log_path = LOG_DIR / f"exp84_download_{int(time.time())}.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {total_new} new, {total_skipped} existed, {total_failed} failed")
    print(f"Wall: {summary['totals']['elapsed_sec']:.0f}s")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
