# Run command:
# python scripts/download_ch_music_archive.py
#
# To customize the URL or output directory:
# python scripts/download_ch_music_archive.py --url "https://roi-chinesemusic.com/en_us/samav-archive/aas-obow/" --output-dir "data/raw/ChMusic_archive"

import os
import sys
import re
import csv
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def sanitize_filename(name: str) -> str:
    """Removes or replaces characters not allowed in file paths."""
    clean = re.sub(r'[\\/*?:"<>|]', '_', name)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:120]


def download_file(url: str, output_path: str, chunk_size: int = 1024 * 64) -> bool:
    """Downloads a file from url to output_path with progress bar."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Skipping (already exists): {os.path.basename(output_path)}")
        return True

    temp_path = output_path + ".tmp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://roi-chinesemusic.com/",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(output_path)[:30],
                leave=False,
            ) as pbar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        os.replace(temp_path, output_path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error downloading {url}: {e}")
        return False


def get_media_url_for_item(item_id: str) -> str:
    """Resolves direct media URL for an archive item ID."""
    # Common pattern for Chinese Music Archive OSS storage:
    # A7... -> 2A07
    if item_id.startswith("A7"):
        return f"https://audio-cml.oss-cn-hangzhou.aliyuncs.com/2A07/{item_id}.mp4"

    # Fallback: scrape video page
    page_url = f"https://roi-chinesemusic.com/en_us/aiovg_videos/{item_id}/"
    req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        matches = re.findall(r'https?://[^\s"\'<>]+\.(?:mp3|mp4|wav|m3u8)', html)
        if matches:
            return matches[0]
    except Exception:
        pass
    return f"https://audio-cml.oss-cn-hangzhou.aliyuncs.com/2A07/{item_id}.mp4"


def download_ch_music_archive(
    archive_url: str = "https://roi-chinesemusic.com/en_us/samav-archive/aas-obow/",
    output_dir: str = "data/raw/ChMusic_archive",
    limit: int = None,
):
    """
    Scrapes and downloads tracks from the Chinese Music Archive (roi-chinesemusic.com).
    Also writes a metadata.csv mapping tracks, performers, and local files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Fetching archive catalog from: {archive_url}")

    req = urllib.request.Request(archive_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    items = []
    for r in rows:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', r, re.DOTALL)
        if not tds:
            continue
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in tds]
        links = re.findall(r'aiovg_videos/([^/]+)/', r)
        if links:
            item_id = links[0]
            title = cells[1] if len(cells) > 1 else item_id
            play_form = cells[4] if len(cells) > 4 else ""
            performers = cells[5] if len(cells) > 5 else ""
            composer = cells[6] if len(cells) > 6 else ""
            items.append({
                "id": item_id,
                "title": title,
                "play_form": play_form,
                "performers": performers,
                "composer": composer,
            })

    print(f"Found {len(items)} items in archive.")
    if limit:
        items = items[:limit]
        print(f"Limiting download to first {limit} items.")

    metadata_records = []
    metadata_path = os.path.join(output_dir, "metadata.csv")

    for idx, item in enumerate(items, 1):
        media_url = get_media_url_for_item(item["id"])
        ext = media_url.split(".")[-1].split("?")[0] if "." in media_url else "mp4"
        safe_title = sanitize_filename(item["title"])
        filename = f"{item['id']}_{safe_title}.{ext}"
        filepath = os.path.join(output_dir, filename)

        print(f"[{idx}/{len(items)}] Downloading: {item['id']} - {item['title']}")
        success = download_file(media_url, filepath)

        metadata_records.append({
            "id": item["id"],
            "title": item["title"],
            "play_form": item["play_form"],
            "performers": item["performers"],
            "composer": item["composer"],
            "media_url": media_url,
            "filename": filename,
            "downloaded": success,
        })

    # Save metadata CSV
    fieldnames = ["id", "title", "play_form", "performers", "composer", "media_url", "filename", "downloaded"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    print(f"\nAll operations completed! Files saved to: {output_dir}")
    print(f"Metadata index saved to: {metadata_path}")


def download_youtube_or_stream(video_url: str, output_dir: str = "downloads"):
    """
    Downloads a video or playlist using yt-dlp.
    Handles YouTube links, raw M3U8 streams, and supported web embeds.
    """
    if yt_dlp is None:
        raise ImportError(
            "yt-dlp is not installed. Please install it using: pip install yt-dlp"
        )

    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['tr', 'en'],
        'embedsubtitles': True,
        'retries': 10,
        'fragment_retries': 10,
        'quiet': False,
        'no_warnings': False,
    }

    print(f"Starting download for: {video_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error during download: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download audio/video from Chinese Music Archive or streaming platforms.")
    parser.add_argument(
        "--url",
        default="https://roi-chinesemusic.com/en_us/samav-archive/aas-obow/",
        help="Archive URL or YouTube/Stream URL to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/ChMusic_archive",
        help="Destination directory for downloaded files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to download (useful for testing).",
    )
    args = parser.parse_args()

    if "roi-chinesemusic.com" in args.url:
        download_ch_music_archive(archive_url=args.url, output_dir=args.output_dir, limit=args.limit)
    else:
        download_youtube_or_stream(video_url=args.url, output_dir=args.output_dir)


if __name__ == "__main__":
    main()