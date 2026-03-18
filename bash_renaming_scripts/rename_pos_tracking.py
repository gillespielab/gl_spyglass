import os
import re
import sys

# NOTE: created using chatgpt

def rename_files_with_suffixes(root_dir, suffixes):
    """
    Recursively scan subdirectories under root_dir.
    For each subdirectory:
      - Skip if any file already contains '_orig'
      - Only rename files that end in _r{integer}.{number}.{suffix}
        by inserting '_orig' before the integer.
    """
    suffix_pattern = "|".join(map(re.escape, suffixes))
    # Require prefix to end with _r{integer}
    pattern = re.compile(rf"^(.*_r\d+)\.(\d+)\.({suffix_pattern})$")

    for dirpath, _, filenames in os.walk(root_dir):
        # Skip the directory entirely if any file has _orig in its name
        if any("_orig" in f for f in filenames):
            print(f"Skipping directory (already has _orig files): {dirpath}")
            continue

        for filename in filenames:
            match = pattern.match(filename)
            if match:
                prefix, number, suffix = match.groups()
                new_name = f"{prefix}_orig.{number}.{suffix}"
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_name)

                # Avoid overwriting
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} → {new_path}")
                else:
                    print(f"Skipped (already exists): {new_path}")
            else:
                # File didn't match _r{integer} rule
                continue

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} /path/to/root_dir suffix1 [suffix2 ...]")
        print(f"Example: python {sys.argv[0]} ./data trackgeometry videoPositionTracking")
        sys.exit(1)

    root_dir = sys.argv[1]
    suffixes = sys.argv[2:]

    rename_files_with_suffixes(root_dir, suffixes)
