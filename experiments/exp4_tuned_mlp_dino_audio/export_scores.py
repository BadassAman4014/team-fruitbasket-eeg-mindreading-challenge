"""Export DINOv2+CLAP score matrices from a finished exp4 run for external blending.

Reads the already-computed raw score matrices from an exp4 run directory and
writes a single portable NPZ in the teammate-export schema used by exp6/exp8:

  heldout_raw_scores   (582, 100)  float32  six-subject 901..1000 query vs candidates
  heldout_query_ids    (582,)      int64    training stimulus ID per query row
  heldout_query_subj   (582,)      int64    subject number per query row
  heldout_cand_ids     (100,)      int64    candidate stimulus IDs 901..1000
  sub_01..06_raw_scores (102, 102) float32  test queries vs 102 candidates per subject
  test_cand_ids        (102,)      int64    candidate stimulus IDs 1001..1102
  seeds                (K,)        int64    training seeds used

Row/column conventions:
  - heldout rows are subject-major (all subject-1 queries, then subject-2, ...).
  - test rows are original recording order (never sorted by candidate ID or label).
  - columns are always the explicit candidate ID arrays above.
  - raw_scores are NEVER CSLS-adjusted; the consumer applies their own correction.

Usage (run from repo root):
    python experiments/exp4_tuned_mlp_dino_audio/export_scores.py \\
        --run-dir experiments/exp4_tuned_mlp_dino_audio/runs/<your-run> \\
        --output path/to/dino_scores.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUBJECTS = tuple(range(1, 7))
HELDOUT_IDS = np.arange(901, 1001, dtype=np.int64)
TEST_IDS = np.arange(1001, 1103, dtype=np.int64)
EXPECTED_HELDOUT_SHAPE = (582, 100)
EXPECTED_TEST_SHAPE = (102, 102)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Completed exp4 run directory (must contain heldout_outputs.npz and sub-XX_test_outputs.npz)"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination .npz (a sibling .json metadata file is written alongside it)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    return args


def require(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def finite(array: np.ndarray, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    if result.shape != expected_shape:
        raise ValueError(f"{name}: expected shape {expected_shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: contains non-finite values")
    return result


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(".json")

    if not args.overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"Refusing to overwrite: {output}")

    # Load and validate run config.
    config_path = require(run_dir / "config.json", "exp4 config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    subjects = tuple(config.get("subjects", ()))
    prediction_subjects = tuple(config.get("prediction_subjects", ()))
    if subjects != SUBJECTS:
        raise ValueError(f"Run must use all 6 subjects; got {subjects}")
    if prediction_subjects != SUBJECTS:
        raise ValueError(f"Run must predict all 6 subjects; got {prediction_subjects}")
    if not config.get("final_full_fit"):
        raise ValueError("Run must use --final-full-fit for test scores to be from the full-fit models")
    seeds = config.get("seeds", [])

    # Load held-out scores.
    heldout_path = require(run_dir / "heldout_outputs.npz", "heldout_outputs.npz")
    with np.load(heldout_path) as pack:
        heldout_raw = finite(pack["raw_scores"], EXPECTED_HELDOUT_SHAPE, "heldout raw_scores")
        query_ids = np.asarray(pack["query_ids"], dtype=np.int64)
        query_subjects = np.asarray(pack["query_subjects"], dtype=np.int64)
        candidate_ids = np.asarray(pack["candidate_ids"], dtype=np.int64)
        if query_ids.shape != (582,):
            raise ValueError(f"heldout query_ids must be (582,), got {query_ids.shape}")
        if not np.array_equal(candidate_ids, HELDOUT_IDS):
            raise ValueError("heldout candidate_ids must be 901..1000")

    # Load test scores per subject.
    arrays: dict[str, np.ndarray] = {
        "heldout_raw_scores": heldout_raw,
        "heldout_query_ids": query_ids,
        "heldout_query_subjects": query_subjects,
        "heldout_candidate_ids": HELDOUT_IDS,
        "test_candidate_ids": TEST_IDS,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    for subject in SUBJECTS:
        test_path = require(
            run_dir / f"sub-{subject:02d}_test_outputs.npz",
            f"sub-{subject:02d}_test_outputs.npz"
        )
        with np.load(test_path) as pack:
            raw = finite(pack["raw_scores"], EXPECTED_TEST_SHAPE, f"sub-{subject:02d} raw_scores")
            test_cands = np.asarray(pack["candidate_ids"], dtype=np.int64)
            row_indices = np.asarray(pack["test_row_indices"], dtype=np.int64)
            if not np.array_equal(test_cands, TEST_IDS):
                raise ValueError(f"sub-{subject:02d} candidate_ids must be 1001..1102")
            if not np.array_equal(row_indices, np.arange(102, dtype=np.int64)):
                raise ValueError(f"sub-{subject:02d} test rows are not in original order")
            arrays[f"sub_{subject:02d}_raw_scores"] = raw

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {
        "schema_version": 1,
        "source_run": str(run_dir),
        "output": str(output),
        "training_performed": False,
        "subjects": list(SUBJECTS),
        "selection_fit_id_range": [1, 900],
        "heldout_id_range": [901, 1000],
        "full_fit_id_range": [1, 1000],
        "seeds": list(seeds),
        "score_source": "raw_scores (never adjusted_scores)",
        "heldout_row_order": "subject-major, ascending stimulus ID within each subject",
        "test_row_order": "original recording order for all subjects",
        "candidate_columns": {
            "heldout": "IDs 901..1000 (100 candidates)",
            "test": "IDs 1001..1102 (102 candidates)"
        },
        "private_labels_read": False,
        "model": config.get("audio_model", "unknown"),
        "note": "DINOv2+CLAP exp4 MLP score matrices for external blending",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved DINOv2 score export: {output}")
    print(f"Saved metadata:            {metadata_path}")
    print(f"heldout=(582,100); test subjects 1..6=(102,102), original row order")
    print(f"Subjects: {subjects} | Seeds: {seeds} | Full-fit: {config.get('final_full_fit')}")


if __name__ == "__main__":
    main()
