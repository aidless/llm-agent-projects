#!/usr/bin/env python3
"""Download NLTK tokenizers once, during `make check-install` (needs network).

`make check` itself stays offline. Exit 0 when `punkt` or `punkt_tab` is
already present or was just downloaded. Exit 1 when neither is available.
"""
from __future__ import annotations

import sys


def _has(name: str) -> bool:
    import nltk

    try:
        nltk.data.find(f"tokenizers/{name}")
        return True
    except LookupError:
        return False


def main() -> int:
    try:
        import nltk
    except ImportError:
        print("nltk is not installed; run: python -m pip install -r requirements-check.txt", file=sys.stderr)
        return 1

    ok = False
    for name in ("punkt", "punkt_tab"):
        if _has(name):
            print(f"nltk {name}: present")
            ok = True
            continue
        try:
            downloaded = nltk.download(name, quiet=True)
        except Exception as exc:  # noqa: BLE001 — installer should report and continue
            print(f"nltk {name}: download error: {exc}", file=sys.stderr)
            continue
        if downloaded or _has(name):
            print(f"nltk {name}: downloaded")
            ok = True
        else:
            print(f"nltk {name}: not available", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
