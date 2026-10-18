"""Phase 1 — Test-set prediction and evaluation.

Loads saved Phase 1 models, extracts features for the 102 test videos,
and scores against tst_stim/ ground truth for sub-01 through sub-04.
Also generates submission .npy files for sub-05 and sub-06.

Usage
-----
    # Phase 1A — ViCLIP
    python experiments/phase1_predict.py --phase 1a --device cuda

    # Phase 1B — ImageBind (concat fusion)
    python experiments/phase1_predict.py --phase 1b --device cuda

    # Phase 1C — Hybrid
    python experiments/phase1_predict.py --phase 1c --device cuda
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.data_utils import load_test_eeg, decimate_eeg, zscore_per_subject
from src.pipeline   import EncodingModel
from src.metrics    import compute_ranks, retrieval_metrics, print_metrics, apply_anti_hubness


# ── Config ───────────────────────────────────────────────────────────────────

TEST_DIR       = "./tst"
STIM_TEST_DIR  = "./stimuli_test"
GT_DIR         = "./tst_stim"          # ground truth for sub-01..04
TEST_IDS       = list(range(1001, 1103))   # 102 test videos
EVAL_SUBJECTS  = [1, 2, 3, 4]         # subjects with ground truth
PRED_SUBJECTS  = [5, 6]               # submission targets


# ── Feature extraction for test videos ───────────────────────────────────────


def video_path(stim_id: int, stim_dir: str) -> str:
    return os.path.join(stim_dir, f"{int(stim_id):04d}.mp4")


def get_test_features_viclip(cache_dir, device, viclip_ckpt=None):
    """Extract or load ViCLIP features for the 102 test videos."""
    cache_path = os.path.join(cache_dir, "viclip_test.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached test features from {cache_path}")
        return np.load(cache_path)

    from src.features.viclip_extractor import ViCLIPExtractor
    extractor = ViCLIPExtractor(ckpt_path=viclip_ckpt, device=device)
    paths = [video_path(sid, STIM_TEST_DIR) for sid in TEST_IDS]
    print(f"  Extracting ViCLIP features for {len(paths)} test videos ...")
    feats = extractor.extract(paths)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, feats)
    print(f"  Cached → {cache_path}")
    return feats


def get_test_features_imagebind(cache_dir, device, fusion="concat"):
    """Extract or load ImageBind features for the 102 test videos."""
    cache_path = os.path.join(cache_dir, f"imagebind_test_{fusion}.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached test features from {cache_path}")
        return np.load(cache_path)

    from src.features.imagebind_extractor import ImageBindExtractor
    extractor = ImageBindExtractor(
        device=device,
        audio_cache_dir=os.path.join(cache_dir, "wav_test"),
    )
    paths = [video_path(sid, STIM_TEST_DIR) for sid in TEST_IDS]
    print(f"  Extracting ImageBind features ({fusion}) for {len(paths)} test videos ...")
    feats = extractor.extract_video_audio(paths, fusion=fusion)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, feats)
    print(f"  Cached → {cache_path}")
    return feats


def get_test_features_hybrid(cache_dir, device, viclip_ckpt=None):
    """Extract or load ViCLIP video + ImageBind audio features for test videos."""
    cache_path = os.path.join(cache_dir, "hybrid_test.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached hybrid test features from {cache_path}")
        return np.load(cache_path)

    from src.features.viclip_extractor import ViCLIPExtractor
    from src.features.imagebind_extractor import ImageBindExtractor

    paths = [video_path(sid, STIM_TEST_DIR) for sid in TEST_IDS]

    viclip_ext = ViCLIPExtractor(ckpt_path=viclip_ckpt, device=device)
    print(f"  Extracting ViCLIP test features for {len(paths)} videos ...")
    V = viclip_ext.extract(paths)

    ib_ext = ImageBindExtractor(
        device=device,
        audio_cache_dir=os.path.join(cache_dir, "wav_test"),
    )
    print(f"  Extracting ImageBind audio test features for {len(paths)} videos ...")
    A = ib_ext.extract_audio(paths)

    feats = np.concatenate([V, A], axis=1)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, feats)
    print(f"  Cached → {cache_path}  shape={feats.shape}")
    return feats


def get_test_features_viclip_ast(cache_dir, device, viclip_ckpt=None):
    """Extract or load ViCLIP video + AST audio features for test videos."""
    cache_path = os.path.join(cache_dir, "viclip_ast_test.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached ViCLIP+AST test features from {cache_path}")
        return np.load(cache_path)

    from src.features.viclip_extractor import ViCLIPExtractor
    from src.features.ast_extractor import ASTExtractor

    paths = [video_path(sid, STIM_TEST_DIR) for sid in TEST_IDS]

    # Reuse ViCLIP test features if already extracted in Phase 1A
    viclip_test_path = os.path.join(cache_dir, "viclip_test.npy")
    alt_path = os.path.join("./cache/viclip", "viclip_test.npy")
    if os.path.exists(viclip_test_path):
        print(f"  Reusing cached ViCLIP test features from {viclip_test_path}")
        V = np.load(viclip_test_path)
    elif os.path.exists(alt_path):
        print(f"  Reusing cached ViCLIP test features from {alt_path}")
        V = np.load(alt_path)
    else:
        viclip_ext = ViCLIPExtractor(ckpt_path=viclip_ckpt, device=device)
        print(f"  Extracting ViCLIP test features for {len(paths)} videos ...")
        V = viclip_ext.extract(paths)

    ast_ext = ASTExtractor(
        device=device,
        audio_cache_dir=os.path.join(cache_dir, "wav_test"),
    )
    print(f"  Extracting AST audio test features for {len(paths)} videos ...")
    A = ast_ext.extract_audio(paths)

    feats = np.concatenate([V, A], axis=1)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, feats)
    print(f"  Cached → {cache_path}  shape={feats.shape}")
    return feats


# ── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_subject(model, test_eeg_raw, gt_stim_ids, test_feats, sub_label, anti_hub="none", csls_k=10):
    """Score a subject against ground truth, matching organizer's evaluation logic.

    The organizer sorts test EEG rows by ascending ground-truth stimulus ID
    so the diagonal of the similarity matrix is the correct match.
    """
    # Preprocess test EEG
    eeg = decimate_eeg(test_eeg_raw)

    # Sort rows by ground truth stimulus ID (organizer does: order = np.argsort(gt))
    order = np.argsort(gt_stim_ids)
    eeg_sorted = eeg[order]

    # Build similarity matrix — rows are sorted EEG trials, columns are sorted test videos
    S = model.similarity_matrix(eeg_sorted, test_feats)   # (102, 102)

    if anti_hub != "none":
        S = apply_anti_hubness(S, method=anti_hub, k=csls_k)

    # Now diagonal = correct match → compute_ranks expects this
    ranks = compute_ranks(S)
    metrics = retrieval_metrics(ranks)
    print_metrics(metrics, label=sub_label)
    return S, ranks, metrics



def generate_submission(model, test_eeg_raw, test_feats, sub_num, out_dir):
    """Generate ranked candidate list .npy for a submission subject."""
    eeg = decimate_eeg(test_eeg_raw)

    # For submission subjects (5, 6), we do NOT sort rows.
    # Similarity: row i = EEG trial i (original order), col j = test video j (sorted 1001..1102)
    S = model.similarity_matrix(eeg, test_feats)   # (102, 102)

    video_ids = np.array(TEST_IDS)              # [1001, 1002, ..., 1102]
    order_idx = np.argsort(-S, axis=1)           # column indices, best-first
    ranked = video_ids[order_idx]                # (102, 102) video IDs, best-first

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sub-{sub_num:02d}_ranked_candidate_list.npy")
    np.save(path, ranked)
    print(f"  Submission saved → {path}  shape={ranked.shape}")
    return ranked


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    # 1. Extract test features based on phase
    print(f"\n{'='*60}")
    print(f"  Phase {args.phase.upper()} — Test Prediction")
    print(f"{'='*60}\n")

    if args.phase == "1a":
        test_feats = get_test_features_viclip(
            args.feat_cache, args.device, args.viclip_ckpt
        )
    elif args.phase == "1b":
        test_feats = get_test_features_imagebind(
            args.feat_cache, args.device, args.fusion
        )
    elif args.phase == "1c":
        test_feats = get_test_features_hybrid(
            args.feat_cache, args.device, args.viclip_ckpt
        )
    elif args.phase == "1d":
        test_feats = get_test_features_viclip_ast(
            args.feat_cache, args.device, args.viclip_ckpt
        )
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    print(f"  Test features shape: {test_feats.shape}\n")

    # 2. Evaluate on sub-01 through sub-04 (subjects with ground truth)
    all_metrics = {}
    os.makedirs(args.out_dir, exist_ok=True)
    video_ids = np.array(TEST_IDS)

    for sub in EVAL_SUBJECTS:
        model_path = os.path.join(args.model_dir, f"phase{args.phase}_sub-{sub:02d}.joblib")
        if not os.path.exists(model_path):
            print(f"  [sub-{sub:02d}] Model not found at {model_path} — skipping")
            continue

        model = EncodingModel.load(model_path)
        eeg_raw = load_test_eeg(os.path.join(TEST_DIR, f"sub-{sub:02d}_tst.npy"))
        gt = np.load(os.path.join(GT_DIR, f"sub-{sub:02d}_tst_stim.npy"))

        S, ranks, metrics = evaluate_subject(
            model, eeg_raw, gt, test_feats, f"sub-{sub:02d} [TEST]",
            anti_hub=args.anti_hub, csls_k=args.csls_k
        )
        all_metrics[f"sub-{sub:02d}"] = metrics

        # Save evaluation npy files for sub-01..04
        np.save(os.path.join(args.out_dir, f"phase{args.phase}_sub-{sub:02d}_ranks.npy"), ranks)
        np.save(os.path.join(args.out_dir, f"phase{args.phase}_sub-{sub:02d}_similarity.npy"), S)
        ranked_cand = video_ids[np.argsort(-S, axis=1)]
        np.save(os.path.join(args.out_dir, f"phase{args.phase}_sub-{sub:02d}_ranked_candidate_list.npy"), ranked_cand)

    print(f"\nSaved sub-01..04 evaluation .npy files (ranks, similarity, candidate lists) to: {args.out_dir}")

    # 3. Print summary
    if all_metrics:
        print(f"\n{'─'*60}")
        print(f"  Phase {args.phase.upper()} — TEST Summary")
        print(f"{'─'*60}")
        avg_top1 = np.mean([m["top1"] for m in all_metrics.values()])
        avg_top5 = np.mean([m["top5"] for m in all_metrics.values()])
        avg_med  = np.mean([m["median_rank"] for m in all_metrics.values()])
        print(f"  Avg top-1: {avg_top1:.4f}  |  Avg top-5: {avg_top5:.4f}  |  Avg median rank: {avg_med:.1f}")

    # 4. Generate submission files for sub-05 and sub-06
    if args.submit:
        print(f"\n{'='*60}")
        print(f"  Generating Submissions")
        print(f"{'='*60}")
        for sub in PRED_SUBJECTS:
            model_path = os.path.join(args.model_dir, f"phase{args.phase}_sub-01.joblib")
            model = EncodingModel.load(model_path)
            eeg_raw = load_test_eeg(os.path.join(TEST_DIR, f"sub-{sub:02d}_tst.npy"))
            generate_submission(model, eeg_raw, test_feats, sub, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 — Test prediction & evaluation")
    parser.add_argument("--phase", required=True, choices=["1a", "1b", "1c", "1d"],
                        help="Which phase to evaluate (1a: ViCLIP, 1b: ImageBind, 1c: Hybrid IB, 1d: ViCLIP+AST)")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--viclip_ckpt", default=None,
                        help="HuggingFace model ID or .pth path for ViCLIP (phase 1a/1c/1d)")
    parser.add_argument("--fusion",      default="concat", choices=["concat", "mean"],
                        help="Fusion mode for ImageBind (phase 1b only)")
    parser.add_argument("--feat_cache",  default="./cache",
                        help="Root cache directory for test features")
    parser.add_argument("--model_dir",   default="./models/phase1a",
                        help="Directory containing trained .joblib models")
    parser.add_argument("--out_dir",     default="./submissions",
                        help="Directory for submission and evaluation .npy files")
    parser.add_argument("--anti_hub",    default="none", choices=["none", "csls", "mean"],
                        help="Anti-hubness method")
    parser.add_argument("--csls_k",      type=int, default=10,
                        help="CSLS neighborhood size")
    parser.add_argument("--submit", action="store_true",
                        help="Also generate submission files for sub-05/06")
    main(parser.parse_args())

