"""Verify downloaded release assets against their published SHA-256 manifest."""
import argparse
import json

from src.common import digest, path


def verify(directory):
    directory = path(directory)
    hashes = json.loads((directory / "SHA256SUMS.json").read_text())
    if not hashes:
        raise ValueError("Release checksum manifest is empty")
    for name, expected in hashes.items():
        if name != path(name).name:
            raise ValueError(f"Expected a release filename, got {name!r}")
        if digest(directory / name) != expected:
            raise ValueError(f"Release asset hash mismatch: {name}")
    print(f"Verified {len(hashes)} release assets in {directory}")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="artifacts/release")
    verify(parser.parse_args().directory)


if __name__ == "__main__":
    main()
