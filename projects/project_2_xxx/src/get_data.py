"""
get_data.py
-----------
Downloads the "Give Me Some Credit" dataset.
No Kaggle account, no API token, no login required.

Usage:
    pip install requests pandas
    python get_data.py
"""

import hashlib
import zipfile
from pathlib import Path

import requests

# ── destination ───────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")

# ── source URLs ───────────────────────────────────────────────────────────────
# GitHub mirror of the original Kaggle competition data.
# These are the same files used in the original competition (150,000 rows).
FILES = {
    "cs-training.csv": (
        "https://github.com/JLZml/Credit-Scoring-Data-Sets"
        "/raw/refs/heads/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv"
    ),
    "cs-test.csv": (
        "https://github.com/JLZml/Credit-Scoring-Data-Sets"
        "/raw/refs/heads/master/3.%20Kaggle/Give%20Me%20Some%20Credit/cs-test.csv"
    ),
}

# ── MD5 checksums — ensures you have exactly the right data ──────────────────
CHECKSUMS = {
    "cs-training.csv": "2ea8ee1a0bc03bd19a82b2c07dc6e9b7",
    "cs-test.csv":     "6d4905bd76a33cd8e48e60773d5b3bb4",
}


# ─────────────────────────────────────────────────────────────────────────────

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    """Stream-download a file with a simple progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):  # 64 KB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
                print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)
    print()  # newline after progress bar


def verify(path: Path, expected_md5: str) -> bool:
    actual = md5(path)
    if actual != expected_md5:
        print(f"  ✗  Checksum FAILED for {path.name}")
        print(f"     expected: {expected_md5}")
        print(f"     got:      {actual}")
        return False
    return True


def already_exists(path: Path, expected_md5: str) -> bool:
    return path.exists() and md5(path) == expected_md5


def summarise(path: Path) -> None:
    """Print basic stats so you can immediately see the data is correct."""
    import pandas as pd
    df = pd.read_csv(path)
    target = "SeriousDlqin2yrs"
    print(f"\n  Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Default rate  : {df[target].mean():.2%}")
    print(f"  Class ratio   : {(df[target]==0).sum()/(df[target]==1).sum():.1f}:1  (good:bad)")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("  Missing values:")
        for col, n in missing.items():
            print(f"    {col:<45} {n:>6,}  ({n/len(df):.1%})")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════")
    print("  Give Me Some Credit — Data Download")
    print("══════════════════════════════════════════════\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        expected = CHECKSUMS.get(filename)

        # Skip if file already present and valid
        if already_exists(dest, expected):
            size_mb = dest.stat().st_size / 1e6
            print(f"  ✓  {filename} already present ({size_mb:.1f} MB) — skipping")
            continue

        print(f"  → Downloading {filename} ...")
        try:
            download_file(url, dest)
        except requests.RequestException as e:
            print(f"  ✗  Download failed: {e}")
            all_ok = False
            continue

        # Verify checksum
        if expected and not verify(dest, expected):
            dest.unlink()   # delete corrupt file
            all_ok = False
            continue

        size_mb = dest.stat().st_size / 1e6
        print(f"  ✓  {filename}  ({size_mb:.1f} MB)")

    print()

    if not all_ok:
        print("⚠️  One or more files failed. Re-run the script to retry.\n")
        return

    # Quick data summary for the training file
    print("── Training data summary ──────────────────────")
    summarise(RAW_DIR / "cs-training.csv")

    print("\n══════════════════════════════════════════════")
    print("  ✅  Data ready in data/raw/")
    print("══════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()