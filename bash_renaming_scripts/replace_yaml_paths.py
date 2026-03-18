#!/usr/bin/env python3
"""
replace_yaml_paths.py

Recursively search for .yml files under a given path and replace occurrences
of the string "/home/gl-willow/" with "/mnt/".

By default the script runs in dry-run mode and prints all files and lines it
would change. To actually apply the changes, pass the --apply flag.

Usage:
    python replace_yaml_paths.py --path /path/to/search [--apply]

The script makes a .bak backup of each file it modifies when --apply is used.
"""

# NOTE: created using claude

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

SEARCH = "/home/gl-willow/"
REPLACE = "/mnt/"


def find_yaml_files(root: Path) -> List[Path]:
    """Return a sorted list of .yml files under root (recursively)."""
    return sorted(root.rglob("*.yml"))


def scan_file_for_replacements(p: Path, search: str = SEARCH) -> List[Tuple[int, str]]:
    """Return a list of (lineno, line) where 'search' appears in file p.

    Lines are returned as their original content (including newline).
    """
    matches = []
    try:
        with p.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                if search in line:
                    matches.append((i, line.rstrip("\n")))
    except UnicodeDecodeError:
        # skip binary or non-utf8 files
        return []
    return matches


def replace_in_file(
    p: Path, search: str = SEARCH, replace: str = REPLACE, create_backup: bool = True
) -> int:
    """Replace all occurrences of search with replace in file p.

    Creates a backup at p.with_suffix(p.suffix + ".bak") before writing if create_backup is True.
    Returns the number of replacements made.
    """
    text = p.read_text(encoding="utf-8")
    count = text.count(search)
    if count == 0:
        return 0

    # create backup if requested
    if create_backup:
        backup = p.with_suffix(p.suffix + ".bak")
        shutil.copy2(p, backup)

    new_text = text.replace(search, replace)
    p.write_text(new_text, encoding="utf-8")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Find and replace '/home/gl-willow/' -> '/mnt/' in .yml files."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path("."),
        help="Root path to search (default: current directory)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes. If omitted the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".yml",
        help="File extension to search for (default: .yml)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups when applying changes.",
    )

    args = parser.parse_args(argv)
    root: Path = args.path.expanduser().resolve()
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(2)

    files = sorted(root.rglob(f"*{ext}"))
    if not files:
        print(f"No '{ext}' files found under {root}")
        return

    total_matches = 0
    total_replacements = 0

    for p in files:
        # Try to compute relative path from cwd; fall back to absolute if outside cwd
        try:
            rel = p.relative_to(Path.cwd())
        except ValueError:
            rel = p
        matches = scan_file_for_replacements(p)
        if not matches:
            continue

        print(f"\nFile: {rel}")
        for lineno, line in matches:
            new_line = line.replace(SEARCH, REPLACE)
            print(f"  Line {lineno}:\n    - {line}\n    + {new_line}")
            total_matches += 1

        if args.apply:
            replaced = replace_in_file(p, create_backup=not args.no_backup)
            total_replacements += replaced
            if not args.no_backup:
                print(
                    f"  -> Applied: {replaced} replacements (backup: {p.with_suffix(p.suffix + '.bak')})"
                )
            else:
                print(f"  -> Applied: {replaced} replacements")

    print("\nSummary:")
    print(f"  Files scanned: {len(files)}")
    print(f"  Lines that would be changed: {total_matches}")
    if args.apply:
        print(f"  Total replacements made: {total_replacements}")
    else:
        print(
            "  Dry-run mode (no files were modified). Run with --apply to write changes."
        )


if __name__ == "__main__":
    main()
