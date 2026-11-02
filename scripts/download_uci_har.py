"""Download UCI HAR into data/ and verify it against a pinned checksum.

The raw dataset is never committed. Usage: python scripts/download_uci_har.py
"""

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)
SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
INNER_ARCHIVE = "UCI HAR Dataset.zip"


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"refusing to extract outside {dest}: {member.filename}")
    archive.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    marker = args.out / "UCI HAR Dataset" / "activity_labels.txt"
    if marker.exists():
        print(f"already present: {marker.parent}")
        return 0

    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=300) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch: got {digest}, expected {SHA256}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        inner = outer.read(INNER_ARCHIVE)
    with zipfile.ZipFile(io.BytesIO(inner)) as archive:
        safe_extract(archive, args.out)
    print(f"extracted to {marker.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
