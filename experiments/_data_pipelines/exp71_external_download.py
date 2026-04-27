#!/usr/bin/env python3
"""exp71 — Limited external data download for v26 weakest classes.

Downloads xeno-canto + iNaturalist audio for the species where v26 fails.
Restricted set to avoid unnecessary bandwidth + storage.

Priority species (from exp45/61/64/70 audits):
  Tier A (v26 AUC < 0.5 on held-out, fundamentally broken):
    47158son06, 47158son17, 326272, 67107, 74113, bafcur1, litnig1
  Tier B (v26 AUC 0.5-0.7, partially failing):
    22967, 47158son10, 47158son25, 24279, 22961, strher2, 65377
  Tier C (n_train_audio < 50, sparse-data Aves):
    purjay1, plcjay1, rutjac1, thlwre1, trsowl, ...

Strategy:
  - xeno-canto for Aves species (rich coverage)
  - iNaturalist for Amphibia/Mammalia/Insecta/Reptilia (better non-Aves)
  - Quality filter: A-rated (xeno-canto) or research-grade (iNat)
  - Cap: max 100 recordings per species (avoid bias toward over-represented)
  - License: keep CC-BY/CC-BY-SA/CC0; flag CC-BY-NC for separate handling
"""
from __future__ import annotations
import json, time, urllib.parse
from pathlib import Path
import pandas as pd
import requests

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
OUT = ROOT / "data" / "external"
OUT.mkdir(exist_ok=True, parents=True)
LOG_DIR = OUT / "_logs"
LOG_DIR.mkdir(exist_ok=True)

# Priority list per audit findings (2026-04-25)
# Focused on species where external data is queryable.
# Excluded: 47158son06/17 (Insecta sonotypes share generic inat_taxon_id 47158
# and have no scientific name, no usable external source).
PRIORITY_SPECIES = {
    "tier_A": ["326272", "67107", "74113", "bafcur1", "litnig1"],
}
ALL_PRIORITY = sum(PRIORITY_SPECIES.values(), [])
print(f"Priority species: {len(ALL_PRIORITY)}")

# Tight limits — minimum necessary for SED retrain on these 7 species
MAX_XC_PER_SPECIES = 30          # was 100
MAX_INAT_PER_SPECIES = 10        # was 20
MAX_FILE_SIZE_MB = 10            # cap individual file
MIN_QUALITY_XC = "A"
RATE_LIMIT_SEC = 1.0
ALLOWED_EXT = {"mp3", "wav", "ogg", "m4a"}
TIMEOUT_QUERY = 30
TIMEOUT_DL = 60


def load_taxonomy():
    tax = pd.read_csv(DATA / "taxonomy.csv")
    return tax


def safe_download(url, out_path, max_size_mb=MAX_FILE_SIZE_MB):
    """Download with size cap + HTTPS-only + size-streaming."""
    if not url.startswith("https://"):
        return False, "non-https URL"
    try:
        r = requests.get(url, timeout=TIMEOUT_DL, stream=True,
                          headers={"User-Agent": "BirdCLEF2026-research"})
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > max_size_mb * 1024 * 1024:
            return False, f"too large ({int(cl)/1e6:.1f}MB)"
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    total += len(chunk)
                    if total > max_size_mb * 1024 * 1024:
                        f.close()
                        out_path.unlink(missing_ok=True)
                        return False, "exceeded cap during download"
                    f.write(chunk)
        return True, total
    except Exception as e:
        out_path.unlink(missing_ok=True)
        return False, str(e)[:100]


def xc_query(scientific_name: str, max_pages: int = 3):
    """Query xeno-canto API for recordings of a species. HTTPS-only."""
    base = "https://xeno-canto.org/api/2/recordings"
    out = []
    for p in range(1, max_pages + 1):
        q = urllib.parse.quote(scientific_name)
        url = f"{base}?query={q}+q:%3E{MIN_QUALITY_XC[0]}&page={p}"
        try:
            r = requests.get(url, timeout=TIMEOUT_QUERY,
                             headers={"User-Agent": "BirdCLEF2026-research"})
            data = r.json()
        except Exception as e:
            print(f"  xc query error for {scientific_name}: {e}")
            return out
        recs = data.get("recordings", [])
        if not recs: break
        out.extend(recs)
        time.sleep(RATE_LIMIT_SEC)
        if len(out) >= MAX_XC_PER_SPECIES:
            return out[:MAX_XC_PER_SPECIES]
        if p >= int(data.get("numPages", 1)):
            break
    return out


def xc_download(rec, out_dir: Path):
    """Download one xeno-canto recording with size cap."""
    rec_id = str(rec.get("id", ""))
    file_url = rec.get("file", "")
    if not (rec_id and rec_id.isdigit() and file_url):
        return None
    # Force HTTPS
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    elif file_url.startswith("http://"):
        file_url = "https://" + file_url[7:]
    out_path = out_dir / f"xc_{rec_id}.mp3"
    if out_path.exists():
        return {"id": rec_id, "path": str(out_path), "skipped": True}
    ok, info = safe_download(file_url, out_path)
    time.sleep(RATE_LIMIT_SEC)
    if not ok:
        print(f"    xc dl fail {rec_id}: {info}")
        return None
    return {
        "id": rec_id, "path": str(out_path),
        "size_b": info,
        "quality": rec.get("q", ""),
        "length_sec": rec.get("length", ""),
        "license": rec.get("lic", ""),
        "type": rec.get("type", ""),
    }


def inat_query(taxon_id: int, max_results: int = MAX_INAT_PER_SPECIES):
    """Query iNaturalist for sound observations."""
    base = "https://api.inaturalist.org/v1/observations"
    out = []
    page = 1
    per_page = 30
    while len(out) < max_results:
        url = (f"{base}?taxon_id={taxon_id}&sounds=true"
               f"&quality_grade=research&per_page={per_page}&page={page}")
        try:
            r = requests.get(url, timeout=TIMEOUT_QUERY,
                             headers={"User-Agent": "BirdCLEF2026-research"})
            data = r.json()
        except Exception as e:
            print(f"  inat query error for {taxon_id}: {e}")
            return out
        results = data.get("results", [])
        if not results: break
        out.extend(results)
        time.sleep(RATE_LIMIT_SEC)
        if len(results) < per_page: break
        page += 1
        if page > 3: break
    return out[:max_results]


def inat_download(obs, out_dir: Path):
    """Download sounds from iNaturalist observation with size cap."""
    sounds = obs.get("sounds", [])
    obs_id = str(obs.get("id", ""))
    saved = []
    for s in sounds[:2]:  # max 2 sounds per observation
        sound_id = str(s.get("id", ""))
        file_url = s.get("file_url", "")
        if not (sound_id and file_url and obs_id.isdigit() and sound_id.isdigit()):
            continue
        # strip query string before extracting extension
        url_no_q = file_url.split("?")[0]
        ext = url_no_q.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXT:
            continue
        out_path = out_dir / f"inat_{obs_id}_{sound_id}.{ext}"
        if out_path.exists():
            saved.append({"id": f"{obs_id}_{sound_id}", "path": str(out_path), "skipped": True})
            continue
        ok, info = safe_download(file_url, out_path)
        time.sleep(RATE_LIMIT_SEC)
        if not ok:
            continue
        saved.append({
            "id": f"{obs_id}_{sound_id}", "path": str(out_path),
            "size_b": info,
            "license": s.get("license_code", ""),
            "type": s.get("attribution", "")[:80],
        })
    return saved


def main():
    tax = load_taxonomy()
    summary = {"xc": [], "inat": []}

    for tier_name, species_list in PRIORITY_SPECIES.items():
        for sp in species_list:
            sp_row = tax[tax.primary_label.astype(str) == sp]
            if not len(sp_row):
                print(f"[skip] {sp}: not in taxonomy")
                continue
            sci = sp_row.iloc[0].scientific_name
            taxon_id = int(sp_row.iloc[0].inat_taxon_id) if pd.notna(sp_row.iloc[0].inat_taxon_id) else None
            taxon_class = sp_row.iloc[0].class_name

            print(f"\n=== [{tier_name}] {sp} ({sci}, {taxon_class}) inat={taxon_id} ===")

            sp_dir = OUT / sp
            sp_dir.mkdir(exist_ok=True)

            # xeno-canto: DISABLED (v2 API deprecated, v3 needs key)
            if False and taxon_class == "Aves":
                print(f"  -> xeno-canto query: {sci}")
                recs = xc_query(sci)
                print(f"     got {len(recs)} recordings (capped {MAX_RECORDINGS_PER_SPECIES})")
                for rec in recs[:MAX_RECORDINGS_PER_SPECIES]:
                    saved = xc_download(rec, sp_dir)
                    if saved:
                        saved.update({"species": sp, "tier": tier_name, "source": "xc"})
                        summary["xc"].append(saved)

            # iNaturalist: priority for non-Aves, also fallback for Aves
            if taxon_id is not None:
                print(f"  -> inat query taxon_id {taxon_id}")
                obss = inat_query(taxon_id)
                print(f"     got {len(obss)} observations with sounds")
                for obs in obss[:MAX_INAT_PER_SPECIES]:
                    saved_list = inat_download(obs, sp_dir)
                    for s in saved_list:
                        s.update({"species": sp, "tier": tier_name, "source": "inat"})
                        summary["inat"].append(s)

            # Print progress
            xc_count = sum(1 for s in summary["xc"] if s.get("species") == sp)
            inat_count = sum(1 for s in summary["inat"] if s.get("species") == sp)
            print(f"  saved: xc={xc_count}, inat={inat_count}")

    # Save metadata log
    with open(LOG_DIR / f"download_{int(time.time())}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nTotal: xc={len(summary['xc'])}, inat={len(summary['inat'])}")
    print(f"Saved metadata to {LOG_DIR}")


if __name__ == "__main__":
    main()
