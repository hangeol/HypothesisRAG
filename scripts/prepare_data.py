#!/usr/bin/env python3
"""Download the pinned MIRAGE evaluation collection to a local, ignored path."""
import hashlib
import json
from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/gzxiong/MIRAGE/392943af99cd94cafd50a0de2e7fca24bbf65494/benchmark.json"
SHA256 = "6f7f08c64cd2efe02a5d0c247229813c90db345d9dd6e3a451b5d24146d0f8fa"
DESTINATION = Path(__file__).resolve().parents[1] / "MIRAGE" / "benchmark.json"


def main():
    if DESTINATION.exists():
        content = DESTINATION.read_bytes()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise SystemExit(
                "An existing benchmark differs from the pinned collection; move it aside before downloading."
            )
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise SystemExit("Benchmark checksum mismatch; no file was written.")
        collection = json.loads(content)
        assert len(collection["medqa"]) == 1273 and len(collection["mmlu"]) == 1089
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        temporary = DESTINATION.with_suffix(".json.tmp")
        temporary.write_bytes(content)
        temporary.replace(DESTINATION)
    print(f"Verified benchmark: {DESTINATION}")
    print(
        "Dataset contents remain local. See THIRD_PARTY_NOTICES.md for sources and terms."
    )


if __name__ == "__main__":
    main()
