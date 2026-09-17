"""
Reads data/silent_clips.txt and deletes every file listed in it.
Skips comment lines (starting with #) and blank lines.

# Preview what will be deleted (safe):
python scripts/delete_silent_clips.py --dry-run

# Actually delete the 170 silent clips:
python scripts/delete_silent_clips.py

"""
import os
import argparse


def delete_silent_clips(txt_path: str, dry_run: bool = False):
    if not os.path.exists(txt_path):
        print(f"Error: '{txt_path}' not found.")
        return

    deleted = 0
    missing = 0
    errors = 0

    with open(txt_path, "r") as f:
        paths = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    print(f"Found {len(paths)} silent clips in {txt_path}")
    if dry_run:
        print("Dry-run mode — no files will be deleted.\n")

    for path in paths:
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            missing += 1
            continue
        try:
            if dry_run:
                print(f"  [DRY-RUN] would delete {path}")
            else:
                os.remove(path)
                print(f"  [DELETED] {path}")
            deleted += 1
        except Exception as e:
            print(f"  [ERROR]   {path}: {e}")
            errors += 1

    print(f"\nDone: {deleted} deleted, {missing} missing, {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete silent clips listed in data/silent_clips.txt"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/silent_clips.txt",
        help="Path to the silent clips list.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without actually removing files.",
    )
    args = parser.parse_args()

    delete_silent_clips(args.input, dry_run=args.dry_run)
