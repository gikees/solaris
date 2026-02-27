#!/usr/bin/env python3
"""Add a new eval dataset from solaris-engine output to solaris for inference.

This script:
1. Symlinks (or copies) the dataset directory into datasets/eval/
2. Generates an eval IDs JSON file in src/data/eval_ids/
3. Creates a Hydra dataset config YAML in config/dataset/
4. Adds the dataset to config/inference.yaml

Usage:
    python add_eval_dataset.py oneStartsAwayEval --source-dir ../solaris-engine/output2/datasets/eval
    python add_eval_dataset.py sameDirectionEval backToBackEval --source-dir /path/to/datasets/eval

The dataset name should match the directory name in solaris-engine (e.g. oneStartsAwayEval).
A solaris-friendly config name is auto-derived (e.g. eval_one_starts_away).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def camel_to_snake(name):
    """Convert camelCase to snake_case, stripping trailing 'Eval'."""
    name = re.sub(r"Eval$", "", name)
    # Insert underscore before uppercase letters
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return s.lower()


def get_video_frame_count(json_path):
    """Get the number of frames from an action JSON file."""
    with open(json_path) as f:
        actions = json.load(f)
    return len(actions)


def generate_eval_ids(dataset_dir, num_frames=257, seed=42):
    """Generate eval IDs for a multiplayer dataset.

    Each entry is [episode_id, bot1_start, bot1_end, bot2_start, bot2_end].
    Start offsets are chosen randomly such that start + num_frames <= video_length.
    """
    import random

    rng = random.Random(seed)

    mp4_files = sorted(dataset_dir.glob("*_Alpha_*_camera.mp4"), key=lambda p: p.name)
    eval_ids = []

    for episode_id, mp4_path in enumerate(mp4_files):
        # Get action JSON for Alpha
        alpha_json = mp4_path.with_name(mp4_path.name.replace("_camera.mp4", ".json"))
        # Get corresponding Bravo file
        bravo_json_name = alpha_json.name.replace("_Alpha_", "_Bravo_")
        bravo_json = alpha_json.with_name(bravo_json_name)

        if not alpha_json.exists() or not bravo_json.exists():
            print(f"  warning: skipping episode {episode_id}, missing JSON files")
            continue

        alpha_len = get_video_frame_count(alpha_json)
        bravo_len = get_video_frame_count(bravo_json)

        if alpha_len < num_frames or bravo_len < num_frames:
            print(
                f"  warning: skipping episode {episode_id}, too short "
                f"(Alpha: {alpha_len}, Bravo: {bravo_len}, need: {num_frames})"
            )
            continue

        bot1_start = rng.randint(0, alpha_len - num_frames)
        bot2_start = rng.randint(0, bravo_len - num_frames)
        eval_ids.append(
            [episode_id, bot1_start, bot1_start + num_frames, bot2_start, bot2_start + num_frames]
        )

    return eval_ids


def add_to_inference_yaml(inference_yaml_path, config_name):
    """Add a dataset entry to inference.yaml if not already present."""
    with open(inference_yaml_path) as f:
        lines = f.readlines()

    entry = f"  - dataset@eval_datasets.{config_name}: {config_name}\n"

    # Check if already present
    if any(config_name in line for line in lines):
        print(f"  inference.yaml already contains {config_name}, skipping")
        return

    # Insert before the `- _self_` line
    insert_idx = None
    for i, line in enumerate(lines):
        if "_self_" in line:
            insert_idx = i
            break

    if insert_idx is None:
        print("  warning: could not find _self_ in inference.yaml, appending")
        lines.append(entry)
    else:
        lines.insert(insert_idx, entry)

    with open(inference_yaml_path, "w") as f:
        f.writelines(lines)


def process_dataset(eval_name, source_dir, solaris_root):
    """Process a single eval dataset."""
    config_name = "eval_" + camel_to_snake(eval_name)
    print(f"\n=== Processing {eval_name} -> {config_name} ===")

    source_path = Path(source_dir) / eval_name
    if not source_path.exists():
        print(f"  error: source directory not found: {source_path}")
        return False

    test_dir = source_path / "test"
    if not test_dir.exists():
        print(f"  error: test/ subdirectory not found in {source_path}")
        return False

    # 1. Symlink dataset
    dest_path = solaris_root / "datasets" / "eval" / eval_name
    if dest_path.exists():
        print(f"  dataset already exists at {dest_path}, skipping symlink")
    else:
        os.symlink(source_path.resolve(), dest_path)
        print(f"  symlinked {dest_path} -> {source_path.resolve()}")

    # 2. Generate eval IDs
    eval_ids_path = solaris_root / "src" / "data" / "eval_ids" / f"eval_ids_{config_name}.json"
    if eval_ids_path.exists():
        print(f"  eval IDs already exist at {eval_ids_path}, skipping")
    else:
        eval_ids = generate_eval_ids(test_dir)
        if not eval_ids:
            print(f"  error: no valid episodes found in {test_dir}")
            return False
        with open(eval_ids_path, "w") as f:
            json.dump(eval_ids, f, indent=2)
        print(f"  generated {len(eval_ids)} eval IDs -> {eval_ids_path}")

    # 3. Create dataset config YAML
    config_path = solaris_root / "config" / "dataset" / f"{config_name}.yaml"
    if config_path.exists():
        print(f"  config already exists at {config_path}, skipping")
    else:
        config_content = f"""train_dataset_name: ""
test_dataset_name: "eval/{eval_name}/test"
class: src.data.dataset.DatasetMultiplayer
name: {config_name}
multiplayer: true
additional_params:
  bot1_name: "Alpha"
  bot2_name: "Bravo"
  shuffle_bots: ${{shuffle_bots}}
"""
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"  created config -> {config_path}")

    # 4. Add to inference.yaml
    inference_yaml = solaris_root / "config" / "inference.yaml"
    add_to_inference_yaml(inference_yaml, config_name)
    print(f"  updated inference.yaml")

    return True


def main():
    parser = argparse.ArgumentParser(description="Add eval datasets from solaris-engine to solaris")
    parser.add_argument("eval_names", nargs="+", help="Eval dataset names (e.g. oneStartsAwayEval)")
    parser.add_argument(
        "--source-dir",
        default="../solaris-engine/output2/datasets/eval",
        help="Path to solaris-engine eval datasets directory",
    )
    parser.add_argument(
        "--solaris-root",
        default=None,
        help="Path to solaris repo root (default: directory containing this script)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=257,
        help="Number of frames per eval segment (default: 257)",
    )
    args = parser.parse_args()

    solaris_root = Path(args.solaris_root) if args.solaris_root else Path(__file__).parent
    source_dir = Path(args.source_dir)

    if not (solaris_root / "config" / "inference.yaml").exists():
        print(f"error: {solaris_root} does not look like the solaris repo root")
        sys.exit(1)

    success = 0
    for name in args.eval_names:
        if process_dataset(name, source_dir, solaris_root):
            success += 1

    print(f"\nDone: {success}/{len(args.eval_names)} datasets added.")


if __name__ == "__main__":
    main()
