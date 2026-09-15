import os
import yt_dlp

def download_dizi(video_url: str, output_dir: str = "dizi_downloads"):
    """
    Downloads a dizi video or full playlist using yt-dlp.
    Handles YouTube links, raw M3U8 streams, and supported web embeds.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        # Template: Series/Video Title.extension
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        
        # Download highest video + highest audio, fallback to best single file
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        
        # Merge streams into an MKV or MP4 container using ffmpeg
        'merge_output_format': 'mp4',
        
        # Subtitles (fetches English and Turkish subtitles if available)
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['tr', 'en'],
        'embedsubtitles': True,
        
        # Network resilience
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

if __name__ == "__main__":
    # Replace with your target URL (YouTube episode link, channel playlist, or .m3u8 stream)
    target_url = "https://www.youtube.com/watch?v=EXAMPLE_ID"
    download_dizi(target_url)