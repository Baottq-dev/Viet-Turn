#!/usr/bin/env python3
"""
Script 01: Download audio từ YouTube/Podcast

Usage:
    python scripts/01_download_audio.py --url "https://youtube.com/..." --output data/raw
    python scripts/01_download_audio.py --playlist "https://youtube.com/playlist?list=..." --output data/raw
    python scripts/01_download_audio.py --file urls.txt --output data/raw
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def check_ytdlp():
    """Kiểm tra yt-dlp đã cài chưa"""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp chưa được cài đặt!")
        print("   Cài đặt: pip install yt-dlp")
        return False


def download_audio(
    url: str, 
    output_dir: str, 
    audio_format: str = "wav",
    max_duration: Optional[int] = None
) -> bool:
    """
    Download audio từ URL.
    
    Args:
        url: YouTube/Podcast URL
        output_dir: Thư mục output
        audio_format: Format audio (wav, mp3, m4a)
        max_duration: Giới hạn độ dài video (giây), None = không giới hạn
    
    Returns:
        True nếu thành công
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "yt-dlp",
        "--extract-audio",
        f"--audio-format={audio_format}",
        "--audio-quality=0",  # Best quality
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        "--no-playlist",  # Chỉ download video đơn lẻ
        "--restrict-filenames",  # Tên file an toàn
    ]
    
    if max_duration:
        cmd.extend(["--match-filter", f"duration<{max_duration}"])
    
    cmd.append(url)
    
    try:
        print(f"📥 Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Done!")
            return True
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def download_playlist(
    playlist_url: str,
    output_dir: str,
    audio_format: str = "wav",
    max_videos: Optional[int] = None,
    max_duration: Optional[int] = 3600  # Max 1 hour per video
) -> int:
    """
    Download tất cả audio từ playlist.
    
    Returns:
        Số video đã download thành công
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "yt-dlp",
        "--extract-audio",
        f"--audio-format={audio_format}",
        "--audio-quality=0",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        "--yes-playlist",
        "--restrict-filenames",
    ]
    
    if max_videos:
        cmd.extend(["--max-downloads", str(max_videos)])
    
    if max_duration:
        cmd.extend(["--match-filter", f"duration<{max_duration}"])
    
    cmd.append(playlist_url)
    
    print(f"📥 Downloading playlist: {playlist_url}")
    try:
        result = subprocess.run(cmd, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_from_file(
    file_path: str,
    output_dir: str,
    audio_format: str = "wav"
) -> tuple:
    """
    Download từ file chứa danh sách URLs.
    
    Returns:
        (success_count, failed_count)
    """
    urls = Path(file_path).read_text().strip().split("\n")
    urls = [u.strip() for u in urls if u.strip() and not u.startswith("#")]
    
    success = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]")
        if download_audio(url, output_dir, audio_format):
            success += 1
        else:
            failed += 1
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download audio từ YouTube/Podcast cho Viet-Turn project"
    )
    
    parser.add_argument(
        "--url", "-u",
        help="URL của video đơn lẻ"
    )
    parser.add_argument(
        "--playlist", "-p",
        help="URL của playlist"
    )
    parser.add_argument(
        "--file", "-f",
        help="File chứa danh sách URLs (mỗi dòng 1 URL)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/raw/youtube",
        help="Thư mục output (default: data/raw/youtube)"
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "mp3", "m4a"],
        help="Audio format (default: wav)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Số video tối đa (cho playlist)"
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=3600,
        help="Độ dài tối đa mỗi video (giây, default: 3600=1h)"
    )
    
    args = parser.parse_args()
    
    # Check yt-dlp
    if not check_ytdlp():
        sys.exit(1)
    
    # Download
    if args.url:
        success = download_audio(args.url, args.output, args.format, args.max_duration)
        sys.exit(0 if success else 1)
    
    elif args.playlist:
        success = download_playlist(
            args.playlist, args.output, args.format, 
            args.max_videos, args.max_duration
        )
        sys.exit(0 if success else 1)
    
    elif args.file:
        success, failed = download_from_file(args.file, args.output, args.format)
        print(f"\n📊 Summary: {success} success, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
    
    else:
        parser.print_help()
        print("\n⚠️  Cần cung cấp --url, --playlist hoặc --file")
        sys.exit(1)


if __name__ == "__main__":
    main()
