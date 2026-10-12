"""
Extract the MILK10k archive locally. Run this ON YOUR OWN MACHINE (not in a sandboxed bridge) —
it's a plain local unzip and will be far faster there than through any remote-file bridge.

Usage:
    python extract_local.py --zip "D:\\Claude\\milk10k\\archive (1).zip" --out "D:\\Claude\\milk10k\\MILK10k_extracted"
"""
import argparse
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to archive (1).zip")
    ap.add_argument("--out", required=True, help="Output directory to extract into")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, out_dir)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} files extracted")
    print("Done. Expected top-level entries: MILK10k_Training_Input, MILK10k_Test_Input, "
          "MILK10k_Training_Metadata.csv, MILK10k_Training_GroundTruth.csv, "
          "MILK10k_Training_Supplement.csv, MILK10k_Test_Metadata.csv")


if __name__ == "__main__":
    main()
