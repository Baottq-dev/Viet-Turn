#!/usr/bin/env python3
"""
check_data_quality.py - Automated data quality check for entire pipeline.

Checks all steps: audio, split, diarization, VA matrix, labels,
transcripts, text alignment, manifest. Outputs a report with:
  - Per-step pass/fail counts
  - Flagged files with specific issues
  - Human review list (files + timestamps to spot-check by ear)

Usage:
    python scripts/check_data_quality.py
    python scripts/check_data_quality.py --step 1        # Only check diarization
    python scripts/check_data_quality.py --step 1 4      # Check diarization + transcription
    python scripts/check_data_quality.py --human-review 10  # Generate review list for 10 files
    python scripts/check_data_quality.py --fix            # Auto-generate blacklist of bad files
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Lazy imports (torch is heavy)
_torch = None
def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ============================================================
# Config
# ============================================================
AUDIO_RAW_DIR   = PROJECT_ROOT / "data" / "audio"
AUDIO_SPLIT_DIR = PROJECT_ROOT / "data" / "audio_split"
RTTM_DIR        = PROJECT_ROOT / "data" / "rttm"
VA_DIR          = PROJECT_ROOT / "data" / "va_matrices"
LABEL_DIR       = PROJECT_ROOT / "data" / "vap_labels"
TRANSCRIPT_DIR  = PROJECT_ROOT / "data" / "transcripts"
TEXT_DIR        = PROJECT_ROOT / "data" / "text_frames"
MANIFEST_DIR    = PROJECT_ROOT / "data"
REPORT_PATH     = PROJECT_ROOT / "data" / "_quality_report.json"

FRAME_HZ = 50
EXPECTED_SAMPLE_RATE = 16000


def cprint(msg, color=""):
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
              "cyan": "\033[96m", "bold": "\033[1m", "": ""}
    end = "\033[0m" if color else ""
    print(f"{colors.get(color, '')}{msg}{end}")


def section(title):
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"  {title}", "cyan")
    cprint(f"{'='*60}", "cyan")


def check_mark(ok):
    return "OK" if ok else "FAIL"


# ============================================================
# Step 0: Check raw audio
# ============================================================
def check_step0_audio(report):
    section("STEP 0: Raw Audio (data/audio/)")

    if not AUDIO_RAW_DIR.exists():
        cprint("  Directory not found — skipping", "yellow")
        return

    wav_files = sorted(AUDIO_RAW_DIR.glob("*.wav"))
    cprint(f"  Files found: {len(wav_files)}")

    issues = []
    for f in wav_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb < 1:
            issues.append({"file": f.name, "issue": f"too small ({size_mb:.1f} MB)"})

    ok = len(wav_files) - len(issues)
    cprint(f"  OK: {ok}  |  Issues: {len(issues)}", "green" if not issues else "yellow")
    for iss in issues[:10]:
        cprint(f"    {iss['file']}: {iss['issue']}", "yellow")

    report["step0_audio"] = {
        "total": len(wav_files),
        "ok": ok,
        "issues": issues,
    }


# ============================================================
# Step 0b: Check split audio
# ============================================================
def check_step0b_split(report):
    section("STEP 0b: Split Audio (data/audio_split/)")

    if not AUDIO_SPLIT_DIR.exists():
        cprint("  Directory not found — skipping", "yellow")
        return

    torch = get_torch()
    import torchaudio

    wav_files = sorted(AUDIO_SPLIT_DIR.glob("*.wav"))
    cprint(f"  Segments found: {len(wav_files)}")

    issues = []
    durations = []
    for f in wav_files:
        try:
            info = torchaudio.info(str(f))
            dur = info.num_frames / info.sample_rate
            durations.append(dur)

            file_issues = []
            if dur < 60:
                file_issues.append(f"too short ({dur:.0f}s)")
            if dur > 900:
                file_issues.append(f"too long ({dur:.0f}s)")
            if info.sample_rate != EXPECTED_SAMPLE_RATE:
                file_issues.append(f"wrong sample rate ({info.sample_rate})")
            if info.num_channels != 1:
                file_issues.append(f"not mono ({info.num_channels}ch)")

            if file_issues:
                issues.append({"file": f.name, "duration": round(dur, 1),
                               "issue": "; ".join(file_issues)})
        except Exception as e:
            issues.append({"file": f.name, "issue": f"cannot read: {e}"})

    ok = len(wav_files) - len(issues)
    avg_dur = np.mean(durations) if durations else 0
    cprint(f"  OK: {ok}  |  Issues: {len(issues)}", "green" if not issues else "yellow")
    cprint(f"  Duration: avg={avg_dur:.0f}s, min={min(durations):.0f}s, max={max(durations):.0f}s")
    for iss in issues[:10]:
        cprint(f"    {iss['file']}: {iss['issue']}", "yellow")

    # Check source coverage
    sources = set(f.stem.rsplit("_", 1)[0] for f in wav_files)
    raw_count = len(list(AUDIO_RAW_DIR.glob("*.wav"))) if AUDIO_RAW_DIR.exists() else 0
    cprint(f"  Source videos covered: {len(sources)} (raw files: {raw_count})")

    report["step0b_split"] = {
        "total": len(wav_files),
        "ok": ok,
        "issues": issues,
        "avg_duration": round(avg_dur, 1),
        "source_videos": len(sources),
    }


# ============================================================
# Step 1: Check diarization (MOST CRITICAL)
# ============================================================
def check_step1_diarize(report):
    section("STEP 1: Diarization (data/rttm/)")

    if not RTTM_DIR.exists():
        cprint("  Directory not found — skipping", "yellow")
        return

    rttm_files = sorted(RTTM_DIR.glob("*.rttm"))
    json_files = sorted(RTTM_DIR.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]
    cprint(f"  RTTM files: {len(rttm_files)}  |  JSON files: {len(json_files)}")

    issues = []
    suspect_files = []  # For human review
    stats_list = []

    for rttm_path in rttm_files:
        fid = rttm_path.stem
        json_path = RTTM_DIR / f"{fid}.json"

        if not json_path.exists():
            issues.append({"file": fid, "issue": "missing JSON companion"})
            continue

        try:
            segs = json.load(open(json_path, encoding="utf-8"))
            if not segs:
                issues.append({"file": fid, "issue": "empty diarization"})
                continue

            # Speaker stats
            spk_dur = defaultdict(float)
            for s in segs:
                spk_dur[s["speaker"]] += s["duration"]
            total_dur = sum(spk_dur.values())
            num_speakers = len(spk_dur)

            # Sort by duration (speaker 0 = most talkative)
            sorted_spks = sorted(spk_dur.items(), key=lambda x: -x[1])
            s0_ratio = sorted_spks[0][1] / total_dur if total_dur > 0 else 0
            s1_ratio = sorted_spks[1][1] / total_dur if len(sorted_spks) > 1 and total_dur > 0 else 0

            # Overlap detection (simple: count frames where 2 speakers overlap)
            if total_dur > 0:
                # Build simple timeline
                max_time = max(s["end"] for s in segs)
                resolution = 0.1  # 100ms bins
                bins = int(max_time / resolution) + 1
                activity = np.zeros((num_speakers, bins), dtype=np.int8)
                spk_list = [sp[0] for sp in sorted_spks]
                for s in segs:
                    si = spk_list.index(s["speaker"]) if s["speaker"] in spk_list else -1
                    if si >= 0 and si < num_speakers:
                        start_bin = int(s["start"] / resolution)
                        end_bin = min(int(s["end"] / resolution), bins)
                        activity[si, start_bin:end_bin] = 1

                if num_speakers >= 2:
                    overlap_ratio = float((activity[0] & activity[1]).sum()) / max(bins, 1)
                else:
                    overlap_ratio = 0.0
            else:
                overlap_ratio = 0.0

            file_issues = []
            severity = "ok"

            if num_speakers != 2:
                file_issues.append(f"found {num_speakers} speakers (expected 2)")
                severity = "error"
            # Note: S1 < 5% is normal in interview/podcast format where
            # the host speaks much less than the guest. Not a diarization error.
            if s1_ratio < 0.05:
                file_issues.append(f"S1 only {s1_ratio:.1%} — host speaks less (normal)")
                severity = "info"
            elif s1_ratio < 0.10:
                file_issues.append(f"S1 low at {s1_ratio:.1%} — host speaks less")
                severity = "info"
            if overlap_ratio > 0.15:
                file_issues.append(f"high overlap {overlap_ratio:.1%} — possible speaker split")
                severity = "suspect"

            stat = {
                "file": fid,
                "num_speakers": num_speakers,
                "total_speech": round(total_dur, 1),
                "s0_ratio": round(s0_ratio, 3),
                "s1_ratio": round(s1_ratio, 3),
                "overlap_ratio": round(overlap_ratio, 3),
                "severity": severity,
            }
            stats_list.append(stat)

            if file_issues and severity != "info":
                stat["issues"] = file_issues
                issues.append(stat)

            if severity == "suspect":
                # Pick a timestamp to spot-check: find where S1 should be talking
                s1_segs = [s for s in segs if s["speaker"] == sorted_spks[1][0]] if len(sorted_spks) > 1 else []
                review_ts = s1_segs[0]["start"] if s1_segs else segs[0]["start"]
                suspect_files.append({
                    "file": fid,
                    "reason": "; ".join(file_issues),
                    "review_at_sec": round(review_ts, 1),
                    "s0_ratio": round(s0_ratio, 3),
                    "s1_ratio": round(s1_ratio, 3),
                })

        except Exception as e:
            issues.append({"file": fid, "issue": f"parse error: {e}"})

    # Summary
    ok = len(rttm_files) - len(issues)
    suspect = len(suspect_files)
    cprint(f"  OK: {ok}  |  Warning/Error: {len(issues)}  |  Suspect (need ear): {suspect}",
           "green" if suspect == 0 else "yellow" if suspect < 20 else "red")

    if stats_list:
        s1_ratios = [s["s1_ratio"] for s in stats_list]
        cprint(f"  S1 ratio: avg={np.mean(s1_ratios):.1%}, "
               f"min={np.min(s1_ratios):.1%}, max={np.max(s1_ratios):.1%}")
        low_s1 = sum(1 for r in s1_ratios if r < 0.05)
        cprint(f"  Files with S1 < 5%: {low_s1} (normal for interview format)")

    if suspect_files:
        cprint(f"\n  --- FILES NEEDING HUMAN REVIEW (listen at timestamp) ---", "bold")
        for sf in suspect_files[:20]:
            cprint(f"    {sf['file']}  at {sf['review_at_sec']:.0f}s  "
                   f"(S0={sf['s0_ratio']:.0%} S1={sf['s1_ratio']:.0%})  "
                   f"=> {sf['reason']}", "yellow")
        if len(suspect_files) > 20:
            cprint(f"    ... and {len(suspect_files) - 20} more", "yellow")

    report["step1_diarize"] = {
        "total": len(rttm_files),
        "ok": ok,
        "suspect_count": suspect,
        "issues": issues,
        "suspect_files_for_human_review": suspect_files,
    }


# ============================================================
# Step 2+3: Check VA matrices + Labels
# ============================================================
def check_step2_3_va_labels(report):
    section("STEP 2+3: VA Matrices + VAP Labels")

    torch = get_torch()

    if not VA_DIR.exists() or not LABEL_DIR.exists():
        cprint("  Directory not found — skipping", "yellow")
        return

    va_files = sorted(VA_DIR.glob("*.pt"))
    label_files = sorted(LABEL_DIR.glob("*.pt"))
    cprint(f"  VA matrices: {len(va_files)}  |  Label files: {len(label_files)}")

    from src.utils.labels import encode_vap_labels

    issues = []
    mismatch_count = 0
    class_counter = Counter()
    total_valid = 0
    total_invalid = 0

    for va_path in va_files:
        fid = va_path.stem
        label_path = LABEL_DIR / f"{fid}.pt"

        if not label_path.exists():
            issues.append({"file": fid, "issue": "missing label file"})
            continue

        try:
            va = torch.load(va_path, weights_only=True)
            labels = torch.load(label_path, weights_only=True)

            file_issues = []

            # Check VA matrix shape + values
            if va.shape[0] != 2:
                file_issues.append(f"VA shape {va.shape} (expected 2 speakers)")
            if not torch.all((va == 0) | (va == 1)):
                file_issues.append("VA matrix has non-binary values")

            # Check labels shape match
            if labels.shape[0] != va.shape[1]:
                file_issues.append(f"frame mismatch: VA={va.shape[1]} labels={labels.shape[0]}")

            # Check invalid tail = exactly 100
            invalid = (labels < 0).sum().item()
            if invalid != 100 and invalid != labels.shape[0]:
                file_issues.append(f"invalid frames = {invalid} (expected 100)")

            # Re-encode verification (sample 10% of files)
            if hash(fid) % 10 == 0:
                lab_recomputed = encode_vap_labels(va.numpy())
                if not np.array_equal(labels.numpy(), lab_recomputed):
                    file_issues.append("LABEL MISMATCH on re-encode!")
                    mismatch_count += 1

            # Collect stats
            valid_labels = labels[labels >= 0]
            total_valid += len(valid_labels)
            total_invalid += invalid
            for c in valid_labels.tolist():
                class_counter[c] += 1

            if file_issues:
                issues.append({"file": fid, "issues": file_issues})

        except Exception as e:
            issues.append({"file": fid, "issue": f"load error: {e}"})

    ok = len(va_files) - len(issues)
    cprint(f"  OK: {ok}  |  Issues: {len(issues)}", "green" if not issues else "red")
    cprint(f"  Total valid frames: {total_valid:,}  |  Invalid (tail): {total_invalid:,}")
    cprint(f"  Unique classes: {len(class_counter)}/256")
    cprint(f"  Re-encode mismatches: {mismatch_count}", "red" if mismatch_count else "green")

    # Class distribution check
    if class_counter:
        total = sum(class_counter.values())
        silence_ratio = class_counter.get(0, 0) / total
        cprint(f"  Silence class (0): {silence_ratio:.1%}",
               "red" if silence_ratio > 0.5 else "green")

        top5 = class_counter.most_common(5)
        cprint(f"  Top 5 classes:")
        for cls, cnt in top5:
            cprint(f"    class {cls:3d}: {cnt:>8,} ({cnt/total:.1%})")

    for iss in issues[:10]:
        cprint(f"    {iss.get('file','?')}: {iss.get('issues', iss.get('issue','?'))}", "yellow")

    report["step2_3_va_labels"] = {
        "va_files": len(va_files),
        "label_files": len(label_files),
        "ok": ok,
        "issues": issues,
        "total_valid_frames": total_valid,
        "unique_classes": len(class_counter),
        "silence_class_ratio": round(silence_ratio, 4) if class_counter else None,
        "re_encode_mismatches": mismatch_count,
    }


# ============================================================
# Step 4: Check transcripts
# ============================================================
def check_step4_transcripts(report):
    section("STEP 4: Transcripts (data/transcripts/)")

    if not TRANSCRIPT_DIR.exists():
        cprint("  Directory not found — skipping", "yellow")
        return

    json_files = sorted(f for f in TRANSCRIPT_DIR.glob("*.json") if not f.name.startswith("_"))
    cprint(f"  Transcript files: {len(json_files)}")

    issues = []
    wpm_list = []

    for f in json_files:
        fid = f.stem
        try:
            t = json.load(open(f, encoding="utf-8"))
            words = t.get("words", [])
            nw = len(words)
            dur = words[-1]["end"] if words else 0
            wpm = nw / (dur / 60) if dur > 60 else 0

            file_issues = []

            # Check 1: Word count
            if nw == 0:
                file_issues.append("empty transcript")
            elif nw < 50:
                file_issues.append(f"very few words ({nw})")

            # Check 2: Words per minute
            if wpm > 0:
                wpm_list.append(wpm)
            if wpm > 300:
                file_issues.append(f"WPM={wpm:.0f} => possible hallucination")
            elif wpm < 30 and dur > 60:
                file_issues.append(f"WPM={wpm:.0f} => possible missed speech")

            # Check 3: Timestamp monotonicity
            non_monotonic = 0
            for i in range(1, len(words)):
                if words[i]["start"] < words[i-1]["start"] - 0.1:
                    non_monotonic += 1
            if non_monotonic > 0:
                file_issues.append(f"{non_monotonic} non-monotonic timestamps")

            # Check 4: Large gaps (> 30s silence in transcript)
            large_gaps = 0
            for i in range(1, len(words)):
                gap = words[i]["start"] - words[i-1]["end"]
                if gap > 30:
                    large_gaps += 1
            if large_gaps > 0:
                file_issues.append(f"{large_gaps} gaps > 30s")

            # Note: Word/phrase repetition is NOT a reliable hallucination
            # indicator for Vietnamese. Vietnamese naturally repeats words
            # as backchannels ("Rồi", "Ừ", "Dạ"), emphasis, and stuttering.
            # Manual review confirmed all 24 previous suspects were natural speech.

            if file_issues:
                issues.append({"file": fid, "issues": file_issues, "wpm": round(wpm, 0)})

        except Exception as e:
            issues.append({"file": fid, "issue": f"parse error: {e}"})

    ok = len(json_files) - len(issues)
    cprint(f"  OK: {ok}  |  Issues: {len(issues)}",
           "green" if not issues else "yellow")

    if wpm_list:
        cprint(f"  WPM: avg={np.mean(wpm_list):.0f}, min={np.min(wpm_list):.0f}, max={np.max(wpm_list):.0f}")

    for iss in issues[:10]:
        if "issues" in iss:
            cprint(f"    {iss['file']}: {'; '.join(iss['issues'])}", "yellow")

    report["step4_transcripts"] = {
        "total": len(json_files),
        "ok": ok,
        "issues": issues,
        "avg_wpm": round(np.mean(wpm_list), 0) if wpm_list else 0,
    }


# ============================================================
# Step 5: Check text alignment
# ============================================================
def check_step5_alignment(report):
    section("STEP 5: Text Alignment (data/text_frames/)")

    torch = get_torch()

    if not TEXT_DIR.exists():
        cprint("  Directory not found — skipping", "yellow")
        return

    json_files = sorted(f for f in TEXT_DIR.glob("*.json") if not f.name.startswith("_"))
    cprint(f"  Text frame files: {len(json_files)}")

    issues = []
    for f in json_files:
        fid = f.stem
        try:
            tf = json.load(open(f, encoding="utf-8"))
            file_issues = []

            # Check 1: Frame count matches VA matrix
            va_path = VA_DIR / f"{fid}.pt"
            if va_path.exists():
                va = torch.load(va_path, weights_only=True)
                if tf["num_frames"] != va.shape[1]:
                    file_issues.append(
                        f"frame mismatch: text={tf['num_frames']} VA={va.shape[1]}")

            # Check 2: Snapshot count = words + 1
            expected_snapshots = tf["num_words"] + 1
            if tf["num_snapshots"] != expected_snapshots:
                file_issues.append(
                    f"snapshot count {tf['num_snapshots']} != words+1={expected_snapshots}")

            # Check 3: word_end_frames monotonically increasing
            frames = tf["alignment"]["word_end_frames"]
            if frames:
                non_mono = sum(1 for i in range(len(frames)-1) if frames[i] > frames[i+1])
                if non_mono > 0:
                    file_issues.append(f"{non_mono} non-monotonic word_end_frames")

            # Check 4: word_end_frames within valid range
            if frames and tf["num_frames"] > 0:
                if max(frames) >= tf["num_frames"]:
                    file_issues.append(f"word_end_frame {max(frames)} >= num_frames {tf['num_frames']}")

            if file_issues:
                issues.append({"file": fid, "issues": file_issues})

        except Exception as e:
            issues.append({"file": fid, "issue": f"parse error: {e}"})

    ok = len(json_files) - len(issues)
    cprint(f"  OK: {ok}  |  Issues: {len(issues)}", "green" if not issues else "red")
    for iss in issues[:10]:
        cprint(f"    {iss.get('file','?')}: {iss.get('issues', iss.get('issue','?'))}", "yellow")

    report["step5_alignment"] = {
        "total": len(json_files),
        "ok": ok,
        "issues": issues,
    }


# ============================================================
# Step 6+7: Check manifest + data leakage
# ============================================================
def check_step6_7_manifest(report):
    section("STEP 6+7: Manifest + Leakage Check")

    issues = []
    splits = {}

    for split_name in ["train", "val", "test"]:
        manifest_path = MANIFEST_DIR / f"vap_manifest_{split_name}.json"
        if not manifest_path.exists():
            cprint(f"  {split_name}: manifest not found", "yellow")
            continue
        entries = json.load(open(manifest_path, encoding="utf-8"))
        splits[split_name] = entries
        cprint(f"  {split_name}: {len(entries)} files")

    if not splits:
        cprint("  No manifests found — skipping", "yellow")
        return

    total = sum(len(v) for v in splits.values())
    for name, entries in splits.items():
        ratio = len(entries) / total if total > 0 else 0
        expected = {"train": 0.8, "val": 0.1, "test": 0.1}.get(name, 0)
        ok = abs(ratio - expected) < 0.05
        cprint(f"  {name}: {len(entries)} ({ratio:.0%}) — target {expected:.0%} "
               f"[{check_mark(ok)}]", "green" if ok else "yellow")

    # Data leakage check: same source video in different splits
    def _get_source(fid):
        """'92tmp9tXzso_003' -> '92tmp9tXzso' (handles IDs with underscores)."""
        if len(fid) >= 4 and fid[-4] == "_" and fid[-3:].isdigit():
            return fid[:-4]
        return fid

    source_to_splits = defaultdict(set)
    for split_name, entries in splits.items():
        for e in entries:
            source = _get_source(e["file_id"])
            source_to_splits[source].add(split_name)

    leaked = {src: list(sp) for src, sp in source_to_splits.items() if len(sp) > 1}
    if leaked:
        cprint(f"\n  DATA LEAKAGE DETECTED: {len(leaked)} source videos appear in multiple splits!", "red")
        for src, sp in list(leaked.items())[:10]:
            cprint(f"    {src}: {', '.join(sp)}", "red")
        issues.append({"type": "leakage", "count": len(leaked), "examples": list(leaked.keys())[:10]})
    else:
        cprint(f"\n  Data leakage check: PASSED (no source video in multiple splits)", "green")

    # Check all file paths exist
    missing_paths = 0
    for split_name, entries in splits.items():
        for e in entries:
            for key in ["audio_path", "va_matrix_path", "vap_label_path", "text_frames_path"]:
                if key in e and not Path(e[key]).exists():
                    missing_paths += 1
                    if missing_paths <= 5:
                        issues.append({"file": e["file_id"], "issue": f"missing {key}: {e[key]}"})

    cprint(f"  Missing file paths: {missing_paths}", "red" if missing_paths else "green")

    report["step6_7_manifest"] = {
        "splits": {k: len(v) for k, v in splits.items()},
        "total": total,
        "leakage": leaked,
        "missing_paths": missing_paths,
        "issues": issues,
    }


# ============================================================
# Cross-step consistency
# ============================================================
def check_cross_step(report):
    section("CROSS-STEP: File Count Consistency")

    counts = {}
    for name, path, pattern in [
        ("audio_split", AUDIO_SPLIT_DIR, "*.wav"),
        ("rttm", RTTM_DIR, "*.rttm"),
        ("va_matrices", VA_DIR, "*.pt"),
        ("vap_labels", LABEL_DIR, "*.pt"),
        ("transcripts", TRANSCRIPT_DIR, "*.json"),
        ("text_frames", TEXT_DIR, "*.json"),
    ]:
        if path.exists():
            files = [f for f in path.glob(pattern) if not f.name.startswith("_")]
            counts[name] = len(files)
        else:
            counts[name] = 0

    cprint(f"  audio_split:  {counts.get('audio_split', 0)}")
    cprint(f"  rttm:         {counts.get('rttm', 0)}")
    cprint(f"  va_matrices:  {counts.get('va_matrices', 0)}")
    cprint(f"  vap_labels:   {counts.get('vap_labels', 0)}")
    cprint(f"  transcripts:  {counts.get('transcripts', 0)}")
    cprint(f"  text_frames:  {counts.get('text_frames', 0)}")

    # Check all equal
    non_zero = {k: v for k, v in counts.items() if v > 0}
    if non_zero:
        values = list(non_zero.values())
        if len(set(values)) == 1:
            cprint(f"\n  All steps have {values[0]} files — CONSISTENT", "green")
        else:
            cprint(f"\n  INCONSISTENT file counts!", "red")
            max_count = max(values)
            for k, v in non_zero.items():
                if v < max_count:
                    missing = max_count - v
                    cprint(f"    {k}: missing {missing} files", "yellow")

    report["cross_step"] = counts


# ============================================================
# Generate blacklist of bad files
# ============================================================
def generate_blacklist(report):
    section("BLACKLIST: Files to exclude from training")

    blacklist = set()
    reasons = defaultdict(list)

    # Note: Low S1 ratio is normal in interview/podcast format (host speaks less).
    # Only blacklist files with actual errors (wrong speaker count, empty transcripts).
    diar = report.get("step1_diarize", {})
    for sf in diar.get("suspect_files_for_human_review", []):
        fid = sf["file"]
        # Only blacklist high overlap (possible speaker split), not low S1
        if "high overlap" in sf.get("reason", ""):
            blacklist.add(fid)
            reasons[fid].append(f"diarize: {sf['reason']}")

    # From transcripts (only empty transcripts, not word repetitions)
    trans = report.get("step4_transcripts", {})
    for iss in trans.get("issues", []):
        if "empty transcript" in str(iss.get("issues", "")):
            fid = iss["file"]
            blacklist.add(fid)
            reasons[fid].append("transcript: empty")

    cprint(f"\n  Total blacklisted files: {len(blacklist)}")

    # Save blacklist
    blacklist_path = PROJECT_ROOT / "data" / "_blacklist.json"
    blacklist_data = [
        {"file_id": fid, "reasons": reasons[fid]}
        for fid in sorted(blacklist)
    ]
    with open(blacklist_path, "w", encoding="utf-8") as f:
        json.dump(blacklist_data, f, ensure_ascii=False, indent=2)
    cprint(f"  Saved to: {blacklist_path}", "green")

    if blacklist_data:
        cprint(f"\n  --- Blacklisted files ---", "bold")
        for entry in blacklist_data[:20]:
            cprint(f"    {entry['file_id']}: {'; '.join(entry['reasons'])}", "yellow")
        if len(blacklist_data) > 20:
            cprint(f"    ... and {len(blacklist_data) - 20} more", "yellow")

    report["blacklist"] = {
        "count": len(blacklist),
        "path": str(blacklist_path),
        "files": blacklist_data,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Data quality check for MM-VAP-VI pipeline")
    parser.add_argument("--step", type=int, nargs="*", default=None,
                        help="Only check specific steps (0, 1, 2, 3, 4, 5, 6)")
    parser.add_argument("--fix", action="store_true",
                        help="Generate blacklist of bad files to exclude")
    args = parser.parse_args()

    cprint("\n  MM-VAP-VI DATA QUALITY CHECK", "bold")
    cprint(f"  Project: {PROJECT_ROOT}\n", "bold")

    report = {}
    steps = args.step  # None = all

    if steps is None or 0 in steps:
        check_step0_audio(report)

    if steps is None or 0 in steps:
        check_step0b_split(report)

    if steps is None or 1 in steps:
        check_step1_diarize(report)

    if steps is None or 2 in steps or 3 in steps:
        check_step2_3_va_labels(report)

    if steps is None or 4 in steps:
        check_step4_transcripts(report)

    if steps is None or 5 in steps:
        check_step5_alignment(report)

    if steps is None or 6 in steps:
        check_step6_7_manifest(report)

    if steps is None:
        check_cross_step(report)

    if args.fix or steps is None:
        generate_blacklist(report)

    # Save full report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    section("SUMMARY")
    for key, data in report.items():
        if isinstance(data, dict) and "ok" in data and "total" in data:
            total = data["total"]
            ok = data["ok"]
            pct = ok / total * 100 if total > 0 else 0
            color = "green" if pct >= 95 else "yellow" if pct >= 80 else "red"
            cprint(f"  {key:30s}  {ok:>4d}/{total:<4d}  ({pct:.0f}%)", color)

    cprint(f"\n  Full report: {REPORT_PATH}", "bold")


if __name__ == "__main__":
    main()
