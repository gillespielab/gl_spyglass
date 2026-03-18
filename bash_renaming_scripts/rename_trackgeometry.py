from pathlib import Path
import argparse

def fix_trackgeometry_suffix(root_dir: str, apply: bool = False):
    root = Path(root_dir)
    for file in root.rglob("*.trackgeometry.trackgeometry"):
        new_name = file.with_name(file.name.replace(".trackgeometry.trackgeometry", ".trackgeometry"))
        if apply:
            print(f"Renaming: {file} -> {new_name}")
            file.rename(new_name)
        else:
            print(f"[DRY-RUN] Would rename: {file} -> {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix duplicate .trackgeometry.trackgeometry suffixes")
    parser.add_argument("folder", help="Root folder to scan")
    parser.add_argument("--apply", action="store_true", help="Actually perform renaming")
    args = parser.parse_args()

    fix_trackgeometry_suffix(args.folder, args.apply)