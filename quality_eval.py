"""
Audio Quality Evaluation for LLVC Voice Conversion

Computes non-intrusive speech quality metrics on converted audio using:
  1. TorchAudio-SQUIM Objective — PESQ, STOI, SI-SDR (no reference needed)
  2. TorchAudio-SQUIM Subjective — MOS with non-matching reference
  3. UTMOS (MOSNet successor) — reference-free MOS prediction via torch.hub

Evaluation modes:
  --mode source_only    Evaluate source files only (baseline quality)
  --mode converted_only Evaluate converted files only
  --mode compare        Evaluate both source and converted, show quality delta

Usage:
    # Evaluate converted audio quality
    python quality_eval.py --converted_dir benchmark_out

    # Compare source vs converted quality
    python quality_eval.py --source_dir test_wavs --converted_dir benchmark_out --mode compare

    # Run voice conversion inline, then evaluate
    python quality_eval.py --source_dir test_wavs --run_conversion \\
        --checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \\
        --config experiments/llvc_hfg/config.json

    # Evaluate with ONNX-converted outputs
    python quality_eval.py --converted_dir onnx_out --mode converted_only
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import glob_audio_files

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")


# ---------------------------------------------------------------------------
# Model Loaders (lazy, cached)
# ---------------------------------------------------------------------------

_squim_obj_model = None
_squim_subj_model = None
_utmos_predictor = None


def get_squim_objective():
    """Load SQUIM objective model (PESQ, STOI, SI-SDR). ~28MB download."""
    global _squim_obj_model
    if _squim_obj_model is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        print("  Loading SQUIM Objective model...")
        _squim_obj_model = SQUIM_OBJECTIVE.get_model()
        _squim_obj_model.eval()
    return _squim_obj_model


def get_squim_subjective():
    """Load SQUIM subjective model (MOS with reference). ~360MB download."""
    global _squim_subj_model
    if _squim_subj_model is None:
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        print("  Loading SQUIM Subjective model...")
        _squim_subj_model = SQUIM_SUBJECTIVE.get_model()
        _squim_subj_model.eval()
    return _squim_subj_model


def get_utmos():
    """Load UTMOS strong predictor via torch.hub. ~392MB download."""
    global _utmos_predictor
    if _utmos_predictor is None:
        print("  Loading UTMOS predictor (torch.hub)...")
        _utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
    return _utmos_predictor


# ---------------------------------------------------------------------------
# Audio Loading
# ---------------------------------------------------------------------------

TARGET_SR = 16000  # SQUIM and UTMOS both expect 16kHz


def load_audio_for_eval(path):
    """Load audio file, convert to mono float32 at 16kHz.

    Returns:
        (waveform, sr): waveform shape [1, T], sr=16000
    """
    waveform, sr = torchaudio.load(path)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform, TARGET_SR


# ---------------------------------------------------------------------------
# Per-File Evaluation
# ---------------------------------------------------------------------------

def evaluate_squim_objective(waveform):
    """Compute PESQ, STOI, SI-SDR using SQUIM Objective (non-intrusive).

    Args:
        waveform: [1, T] tensor at 16kHz

    Returns:
        dict with stoi, pesq, si_sdr
    """
    model = get_squim_objective()
    with torch.no_grad():
        stoi, pesq, si_sdr = model(waveform)
    return {
        "stoi": round(stoi.item(), 4),
        "pesq": round(pesq.item(), 4),
        "si_sdr": round(si_sdr.item(), 2),
    }


def evaluate_squim_subjective(waveform, reference):
    """Compute MOS using SQUIM Subjective with non-matching reference.

    Args:
        waveform: [1, T] tensor at 16kHz (audio to evaluate)
        reference: [1, T_ref] tensor at 16kHz (any clean speech)

    Returns:
        dict with mos_squim
    """
    model = get_squim_subjective()
    with torch.no_grad():
        mos = model(waveform, reference)
    return {
        "mos_squim": round(mos.item(), 4),
    }


def evaluate_utmos(waveform, sr=TARGET_SR):
    """Compute MOS using UTMOS (reference-free MOSNet successor).

    Args:
        waveform: [1, T] tensor at any sample rate
        sr: sample rate

    Returns:
        dict with mos_utmos
    """
    predictor = get_utmos()
    with torch.no_grad():
        score = predictor(waveform, sr)
    return {
        "mos_utmos": round(score.item(), 4),
    }


def evaluate_file(path, reference_waveform=None):
    """Run all quality metrics on a single audio file.

    Args:
        path: path to audio file
        reference_waveform: optional [1, T] reference for SQUIM subjective

    Returns:
        dict with all metrics
    """
    waveform, sr = load_audio_for_eval(path)
    duration = waveform.shape[-1] / sr

    results = {
        "file": os.path.basename(path),
        "duration_s": round(duration, 2),
    }

    # SQUIM Objective: PESQ, STOI, SI-SDR
    t0 = time.perf_counter()
    obj_metrics = evaluate_squim_objective(waveform)
    results.update(obj_metrics)
    results["squim_obj_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # SQUIM Subjective: MOS (needs any clean reference)
    if reference_waveform is not None:
        t0 = time.perf_counter()
        subj_metrics = evaluate_squim_subjective(waveform, reference_waveform)
        results.update(subj_metrics)
        results["squim_subj_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # UTMOS: reference-free MOS
    t0 = time.perf_counter()
    utmos_metrics = evaluate_utmos(waveform, sr)
    results.update(utmos_metrics)
    results["utmos_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return results


# ---------------------------------------------------------------------------
# Voice Conversion (optional inline conversion)
# ---------------------------------------------------------------------------

def run_voice_conversion(source_dir, output_dir, checkpoint_path, config_path,
                         chunk_factor=4, stream=True):
    """Convert source audio files using LLVC and save to output_dir.

    Returns:
        list of output file paths
    """
    from infer import load_model, load_audio, do_infer, save_audio

    print(f"\n  Running voice conversion...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config:     {config_path}")
    print(f"  Mode:       {'streaming' if stream else 'non-streaming'} (chunk_factor={chunk_factor})")

    model, sr = load_model(
        os.path.join(PROJECT_ROOT, checkpoint_path),
        os.path.join(PROJECT_ROOT, config_path),
    )
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    source_files = sorted(glob_audio_files(source_dir))
    output_files = []

    for fpath in source_files:
        audio = load_audio(fpath, sr)
        with torch.no_grad():
            out, _, _ = do_infer(model, audio, chunk_factor, sr, stream)
        out_path = os.path.join(output_dir, os.path.basename(fpath))
        save_audio(out, out_path, sr)
        output_files.append(out_path)
        print(f"    Converted: {os.path.basename(fpath)}")

    print(f"  Saved {len(output_files)} files to {output_dir}")
    return output_files


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------

def evaluate_directory(audio_dir, label, reference_waveform=None):
    """Evaluate all audio files in a directory.

    Args:
        audio_dir: directory with .wav files
        label: label for this set (e.g., "source", "converted")
        reference_waveform: optional reference for SQUIM subjective

    Returns:
        list of per-file result dicts
    """
    files = sorted(glob_audio_files(audio_dir))
    if not files:
        print(f"  No audio files found in {audio_dir}")
        return []

    print(f"\n  Evaluating {len(files)} files from: {audio_dir}")
    results = []
    for fpath in files:
        result = evaluate_file(fpath, reference_waveform)
        result["set"] = label
        results.append(result)
        # Progress output
        mos_str = f"UTMOS={result['mos_utmos']:.2f}"
        if "mos_squim" in result:
            mos_str += f"  SQUIM-MOS={result['mos_squim']:.2f}"
        print(f"    {result['file']:40s} PESQ={result['pesq']:.2f}  "
              f"STOI={result['stoi']:.4f}  SI-SDR={result['si_sdr']:6.1f}dB  "
              f"{mos_str}")

    return results


def compute_aggregate(results):
    """Compute aggregate statistics from per-file results."""
    if not results:
        return {}

    metrics = ["pesq", "stoi", "si_sdr", "mos_utmos"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results if m in r]
        if vals:
            agg[f"{m}_mean"] = round(np.mean(vals), 4)
            agg[f"{m}_std"] = round(np.std(vals), 4)
            agg[f"{m}_min"] = round(np.min(vals), 4)
            agg[f"{m}_max"] = round(np.max(vals), 4)

    # SQUIM subjective MOS (if available)
    mos_squim_vals = [r["mos_squim"] for r in results if "mos_squim" in r]
    if mos_squim_vals:
        agg["mos_squim_mean"] = round(np.mean(mos_squim_vals), 4)
        agg["mos_squim_std"] = round(np.std(mos_squim_vals), 4)

    agg["num_files"] = len(results)
    return agg


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def print_table(headers, rows):
    col_widths = []
    for i, h in enumerate(headers):
        max_data = max((len(str(r[i])) for r in rows), default=0)
        col_widths.append(max(len(str(h)), max_data))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def print_aggregate(label, agg):
    """Print aggregate metrics."""
    print(f"\n  {label}:")
    print(f"    PESQ:        {agg.get('pesq_mean', 'N/A')} "
          f"(std={agg.get('pesq_std', 'N/A')}, "
          f"range={agg.get('pesq_min', 'N/A')}–{agg.get('pesq_max', 'N/A')})")
    print(f"    STOI:        {agg.get('stoi_mean', 'N/A')} "
          f"(std={agg.get('stoi_std', 'N/A')})")
    print(f"    SI-SDR:      {agg.get('si_sdr_mean', 'N/A')} dB "
          f"(std={agg.get('si_sdr_std', 'N/A')})")
    print(f"    MOS (UTMOS): {agg.get('mos_utmos_mean', 'N/A')} "
          f"(std={agg.get('mos_utmos_std', 'N/A')})")
    if "mos_squim_mean" in agg:
        print(f"    MOS (SQUIM): {agg.get('mos_squim_mean', 'N/A')} "
              f"(std={agg.get('mos_squim_std', 'N/A')})")


def print_comparison(source_agg, converted_agg):
    """Print side-by-side comparison of source vs converted quality."""
    metrics = [
        ("PESQ", "pesq_mean", "higher=better", "1.0–4.5"),
        ("STOI", "stoi_mean", "higher=better", "0.0–1.0"),
        ("SI-SDR", "si_sdr_mean", "higher=better", "dB"),
        ("MOS (UTMOS)", "mos_utmos_mean", "higher=better", "1.0–5.0"),
        ("MOS (SQUIM)", "mos_squim_mean", "higher=better", "1.0–5.0"),
    ]

    rows = []
    for label, key, direction, scale in metrics:
        src_val = source_agg.get(key, None)
        conv_val = converted_agg.get(key, None)
        if src_val is not None and conv_val is not None:
            delta = conv_val - src_val
            delta_str = f"{delta:+.4f}"
        elif conv_val is not None:
            delta_str = "N/A"
        else:
            continue
        rows.append([
            label,
            f"{src_val:.4f}" if src_val is not None else "N/A",
            f"{conv_val:.4f}" if conv_val is not None else "N/A",
            delta_str,
            scale,
        ])

    if rows:
        print_table(
            ["Metric", "Source", "Converted", "Delta", "Scale"],
            rows,
        )


# ---------------------------------------------------------------------------
# Quality Thresholds Reference
# ---------------------------------------------------------------------------

QUALITY_THRESHOLDS = """
  Quality Reference Thresholds:
  ┌────────────┬────────────┬────────────┬────────────────────────────────┐
  │ Metric     │ Poor       │ Fair       │ Good                           │
  ├────────────┼────────────┼────────────┼────────────────────────────────┤
  │ PESQ       │ < 2.0      │ 2.0 – 3.0  │ > 3.0                         │
  │ STOI       │ < 0.75     │ 0.75 – 0.9 │ > 0.9                         │
  │ SI-SDR     │ < 5 dB     │ 5 – 15 dB  │ > 15 dB                       │
  │ MOS (UTMOS)│ < 2.5      │ 2.5 – 3.5  │ > 3.5                         │
  │ MOS (SQUIM)│ < 2.5      │ 2.5 – 3.5  │ > 3.5                         │
  └────────────┴────────────┴────────────┴────────────────────────────────┘
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audio Quality Evaluation for LLVC Voice Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics computed:
  PESQ    Perceptual Evaluation of Speech Quality (1.0–4.5, higher=better)
  STOI    Short-Time Objective Intelligibility (0.0–1.0, higher=better)
  SI-SDR  Scale-Invariant Signal-to-Distortion Ratio (dB, higher=better)
  MOS     Mean Opinion Score via UTMOS (1.0–5.0, higher=better)
  MOS     Mean Opinion Score via SQUIM Subjective (1.0–5.0, with reference)

All metrics are non-intrusive (no clean reference required) except SQUIM
Subjective MOS which uses a non-matching reference signal.
        """,
    )
    parser.add_argument("--source_dir", default="test_wavs",
                        help="Directory with source audio files")
    parser.add_argument("--converted_dir", default=None,
                        help="Directory with converted audio files")
    parser.add_argument("--mode", default="compare",
                        choices=["source_only", "converted_only", "compare"],
                        help="Evaluation mode (default: compare)")
    parser.add_argument("--output_dir", default="quality_eval_out",
                        help="Output directory for JSON report")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max files to evaluate (default: all)")

    # Inline conversion options
    parser.add_argument("--run_conversion", action="store_true",
                        help="Run LLVC voice conversion before evaluation")
    parser.add_argument("--checkpoint",
                        default="llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth",
                        help="Checkpoint for inline conversion")
    parser.add_argument("--config",
                        default="experiments/llvc_hfg/config.json",
                        help="Config for inline conversion")
    parser.add_argument("--chunk_factor", type=int, default=4,
                        help="Chunk factor for streaming conversion (default: 4)")
    parser.add_argument("--no_stream", action="store_true",
                        help="Use non-streaming inference for conversion")

    # Metric toggles
    parser.add_argument("--skip_squim_subj", action="store_true",
                        help="Skip SQUIM subjective MOS (saves ~360MB download)")
    parser.add_argument("--skip_utmos", action="store_true",
                        help="Skip UTMOS MOS (saves ~392MB download)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {}

    print_section("Audio Quality Evaluation — LLVC Voice Conversion")
    print(f"  Source directory:    {args.source_dir}")
    print(f"  Converted directory: {args.converted_dir or '(will be generated)'}")
    print(f"  Mode:               {args.mode}")

    # ------------------------------------------------------------------
    # Step 1: Run inline conversion if requested
    # ------------------------------------------------------------------
    if args.run_conversion:
        conv_out_dir = os.path.join(args.output_dir, "converted")
        run_voice_conversion(
            args.source_dir, conv_out_dir,
            args.checkpoint, args.config,
            args.chunk_factor, stream=not args.no_stream,
        )
        args.converted_dir = conv_out_dir

    # Validate directories
    if args.mode in ("converted_only", "compare") and not args.converted_dir:
        print("\n  ERROR: --converted_dir is required for this mode.")
        print("  Use --run_conversion to generate converted audio inline.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Load models
    # ------------------------------------------------------------------
    print_section("Loading Evaluation Models")

    t0 = time.perf_counter()
    # Always load SQUIM Objective
    get_squim_objective()

    # Load reference for SQUIM Subjective (use first source file as reference)
    reference_waveform = None
    if not args.skip_squim_subj:
        get_squim_subjective()
        # Use the first source file as non-matching reference
        source_files = sorted(glob_audio_files(args.source_dir))
        if source_files:
            reference_waveform, _ = load_audio_for_eval(source_files[0])
            print(f"  SQUIM Subjective reference: {os.path.basename(source_files[0])}")

    if not args.skip_utmos:
        get_utmos()

    model_load_time = time.perf_counter() - t0
    print(f"  Models loaded in {model_load_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Evaluate source files
    # ------------------------------------------------------------------
    source_results = []
    source_agg = {}

    if args.mode in ("source_only", "compare"):
        print_section("Source Audio Quality")
        source_results = evaluate_directory(
            args.source_dir, "source", reference_waveform)
        if args.max_files:
            source_results = source_results[:args.max_files]
        source_agg = compute_aggregate(source_results)
        print_aggregate("Source Aggregate", source_agg)
        report["source"] = {"per_file": source_results, "aggregate": source_agg}

    # ------------------------------------------------------------------
    # Step 4: Evaluate converted files
    # ------------------------------------------------------------------
    converted_results = []
    converted_agg = {}

    if args.mode in ("converted_only", "compare"):
        print_section("Converted Audio Quality")
        converted_results = evaluate_directory(
            args.converted_dir, "converted", reference_waveform)
        if args.max_files:
            converted_results = converted_results[:args.max_files]
        converted_agg = compute_aggregate(converted_results)
        print_aggregate("Converted Aggregate", converted_agg)
        report["converted"] = {
            "per_file": converted_results, "aggregate": converted_agg}

    # ------------------------------------------------------------------
    # Step 5: Comparison
    # ------------------------------------------------------------------
    if args.mode == "compare" and source_agg and converted_agg:
        print_section("Quality Comparison: Source vs Converted")
        print_comparison(source_agg, converted_agg)

        # Per-file comparison (match by filename)
        source_by_name = {r["file"]: r for r in source_results}
        converted_by_name = {}
        for r in converted_results:
            # Handle prefix patterns like "stream_filename.wav"
            fname = r["file"]
            # Strip common prefixes from conversion output
            for prefix in ("stream_", "nonstream_", "converted_"):
                if fname.startswith(prefix):
                    fname = fname[len(prefix):]
                    break
            converted_by_name[fname] = r

        matched = set(source_by_name.keys()) & set(converted_by_name.keys())
        if matched:
            print(f"\n  Per-file comparison ({len(matched)} matched pairs):")
            rows = []
            for fname in sorted(matched):
                s = source_by_name[fname]
                c = converted_by_name[fname]
                rows.append([
                    fname[:30],
                    f"{s['pesq']:.2f} → {c['pesq']:.2f}",
                    f"{s['stoi']:.4f} → {c['stoi']:.4f}",
                    f"{s['si_sdr']:.1f} → {c['si_sdr']:.1f}",
                    f"{s['mos_utmos']:.2f} → {c['mos_utmos']:.2f}",
                ])
            print_table(
                ["File", "PESQ", "STOI", "SI-SDR(dB)", "MOS(UTMOS)"],
                rows,
            )

    # ------------------------------------------------------------------
    # Step 6: Quality thresholds reference
    # ------------------------------------------------------------------
    print_section("Quality Reference Thresholds")
    print(QUALITY_THRESHOLDS)

    # ------------------------------------------------------------------
    # Step 7: Save report
    # ------------------------------------------------------------------
    report_path = os.path.join(args.output_dir, "quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
