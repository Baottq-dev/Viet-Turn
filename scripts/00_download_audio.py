#!/usr/bin/env python3
"""
00_download_audio.py - Download and prepare audio from YouTube

Downloads videos from scripts/urls.txt, extracts audio, and converts
to WAV 16kHz mono format ready for the data pipeline.

Requirements:
    - yt-dlp (pip install yt-dlp)
    - ffmpeg (must be in PATH)

Usage:
    python scripts/00_download_audio.py
    python scripts/00_download_audio.py --urls scripts/urls.txt --output data/audio
    python scripts/00_download_audio.py --max-duration 3600  # skip videos > 1h
    python scripts/00_download_audio.py --dry-run             # show what would be downloaded
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


SAMPLE_RATE = 16000
CHANNELS = 1  # mono
FFMPEG_EXE = None  # Full path to ffmpeg binary, set by check_dependencies


def _ffmpeg_works(exe_path: str) -> bool:
    """Check if an ffmpeg binary actually runs."""
    try:
        result = subprocess.run(
            [exe_path, "-version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0 and len(result.stdout) > 0
    except Exception:
        return False


def find_ffmpeg_exe() -> Optional[str]:
    """Find a working ffmpeg executable (full path)."""
    candidates = []

    # 1. imageio-ffmpeg (bundled static binary, most reliable on Windows)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            candidates.append(exe)
    except ImportError:
        pass

    # 2. PATH
    path = shutil.which("ffmpeg")
    if path:
        candidates.append(path)

    # 3. Conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for subdir in ["Library/bin", "bin"]:
            c = Path(conda_prefix) / subdir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
            if c.exists():
                candidates.append(str(c))

    # Return first one that actually works
    for exe in candidates:
        if _ffmpeg_works(exe):
            return exe

    return None


def check_dependencies(need_ffmpeg: bool = True):
    """Check that yt-dlp and ffmpeg are available."""
    global FFMPEG_EXE
    missing = []
    if shutil.which("yt-dlp") is None:
        missing.append("yt-dlp")
    if need_ffmpeg:
        FFMPEG_EXE = find_ffmpeg_exe()
        if FFMPEG_EXE is None:
            missing.append("ffmpeg")
        else:
            print(f"ffmpeg found: {FFMPEG_EXE}")
    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print("Install with:")
        if "yt-dlp" in missing:
            print("  pip install yt-dlp")
        if "ffmpeg" in missing:
            print("  pip install imageio-ffmpeg")
        sys.exit(1)


def parse_urls(urls_path: Path) -> List[str]:
    """Parse URLs from file, skipping comments and blank lines."""
    urls = []
    with open(urls_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def get_video_info(url: str) -> Optional[dict]:
    """Get video metadata without downloading."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--js-runtimes", "node",
                "--remote-components", "ejs:github",
                "--no-download",
                "--print", "%(id)s",
                "--print", "%(title)s",
                "--print", "%(duration)s",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  [WARN] Cannot get info for {url}: {result.stderr.strip()}")
            return None
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 3:
            return {
                "id": lines[0],
                "title": lines[1],
                "duration": int(float(lines[2])) if lines[2] != "NA" else 0,
            }
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  [WARN] Error getting info for {url}: {e}")
    return None


def format_duration(seconds: int) -> str:
    """Format seconds as H:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def download_and_convert(
    url: str,
    output_dir: Path,
    max_duration: Optional[int] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    """
    Download a single video and convert to WAV 16kHz mono.

    Returns path to the output WAV file, or None on failure.
    """
    # Get video info first
    info = get_video_info(url)
    if info is None:
        return None

    video_id = info["id"]
    title = info["title"]
    duration = info["duration"]
    output_path = output_dir / f"{video_id}.wav"

    # Skip if already downloaded
    if output_path.exists():
        print(f"  [SKIP] {video_id} already exists ({output_path.name})")
        return output_path

    # Skip if too long
    if max_duration and duration > max_duration:
        print(f"  [SKIP] {video_id} too long ({format_duration(duration)} > {format_duration(max_duration)})")
        return None

    print(f"  [DOWN] {video_id} - {title} ({format_duration(duration)})")

    if dry_run:
        return None

    # Step 1: Download best audio with yt-dlp (no ffmpeg needed)
    raw_template = str(output_dir / f"{video_id}.%(ext)s")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--js-runtimes", "node",
                "--remote-components", "ejs:github",
                "-f", "bestaudio",
                "-o", raw_template,
                "--no-playlist",
                "--no-post-overwrites",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,  # 10 min timeout per video
        )
        if result.returncode != 0:
            print(f"  [FAIL] {video_id}: {result.stderr.strip()[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {video_id}: download timed out (>10 min)")
        return None
    except Exception as e:
        print(f"  [FAIL] {video_id}: {e}")
        return None

    # Find the downloaded raw file (could be .webm, .m4a, .opus, etc.)
    raw_files = list(output_dir.glob(f"{video_id}.*"))
    raw_files = [f for f in raw_files if f.suffix != ".wav"]
    if not raw_files:
        print(f"  [FAIL] {video_id}: raw download not found")
        return None
    raw_path = raw_files[0]

    # Step 2: Convert to WAV 16kHz mono using ffmpeg
    try:
        result = subprocess.run(
            [
                FFMPEG_EXE,
                "-i", str(raw_path),
                "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS),
                "-y",  # overwrite
                str(output_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  [FAIL] {video_id}: ffmpeg convert failed: {result.stderr.strip()[:200]}")
            return None
    except Exception as e:
        print(f"  [FAIL] {video_id}: ffmpeg error: {e}")
        return None
    finally:
        # Clean up raw file
        if raw_path.exists():
            raw_path.unlink()

    # Verify output
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [ OK ] {output_path.name} ({size_mb:.1f} MB)")
        return output_path
    else:
        print(f"  [FAIL] {video_id}: output file not found")
        return None


def verify_audio(audio_path: Path) -> dict:
    """Verify audio file format using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            return {
                "sample_rate": int(stream.get("sample_rate", 0)),
                "channels": int(stream.get("channels", 0)),
                "duration": float(stream.get("duration", 0)),
                "codec": stream.get("codec_name", "unknown"),
            }
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and convert to WAV 16kHz mono"
    )
    parser.add_argument(
        "--urls", type=str, default="scripts/urls.txt",
        help="Path to URL list file (default: scripts/urls.txt)",
    )
    parser.add_argument(
        "--output", type=str, default="data/audio",
        help="Output directory for WAV files (default: data/audio)",
    )
    parser.add_argument(
        "--max-duration", type=int, default=None,
        help="Skip videos longer than N seconds (default: no limit)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify all existing audio files in output directory",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[1]
    urls_path = Path(args.urls)
    if not urls_path.is_absolute():
        urls_path = project_root / urls_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Check dependencies (dry-run only needs yt-dlp, verify only needs ffprobe)
    need_ffmpeg = not args.dry_run
    check_dependencies(need_ffmpeg=need_ffmpeg)

    # Verify mode
    if args.verify:
        print(f"=== Verifying audio files in {output_dir} ===\n")
        wav_files = sorted(output_dir.glob("*.wav"))
        if not wav_files:
            print("No WAV files found.")
            return
        for wav in wav_files:
            info = verify_audio(wav)
            sr = info.get("sample_rate", 0)
            ch = info.get("channels", 0)
            dur = info.get("duration", 0)
            status = "OK" if sr == SAMPLE_RATE and ch == CHANNELS else "BAD"
            print(f"  [{status}] {wav.name}: {sr}Hz, {ch}ch, {format_duration(int(dur))}")
            if status == "BAD":
                print(f"         Expected: {SAMPLE_RATE}Hz, {CHANNELS}ch")
        return

    # Parse URLs
    if not urls_path.exists():
        print(f"[ERROR] URL file not found: {urls_path}")
        sys.exit(1)

    urls = parse_urls(urls_path)
    print(f"=== Download Audio for MM-VAP-VI ===\n")
    print(f"URL file:   {urls_path}")
    print(f"Output dir: {output_dir}")
    print(f"Videos:     {len(urls)}")
    if args.max_duration:
        print(f"Max duration: {format_duration(args.max_duration)}")
    if args.dry_run:
        print(f"Mode:       DRY RUN (no downloads)")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download each URL
    success = 0
    skipped = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        result = download_and_convert(
            url, output_dir,
            max_duration=args.max_duration,
            dry_run=args.dry_run,
        )
        if result is not None:
            success += 1
        elif args.dry_run:
            skipped += 1
        else:
            failed += 1

    # Summary
    print(f"\n=== Done ===")
    print(f"Success: {success}")
    if skipped:
        print(f"Skipped: {skipped}")
    if failed:
        print(f"Failed:  {failed}")
    print(f"Output:  {output_dir}")

    if success > 0 and not args.dry_run:
        print(f"\nNext step: python scripts/01_diarize.py --input {output_dir} --output data/rttm")


if __name__ == "__main__":
    main()
