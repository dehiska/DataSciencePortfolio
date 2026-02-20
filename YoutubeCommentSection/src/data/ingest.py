"""
M1 Ingestion Script — collect 10,000 YouTube comments locally.

Usage:
    python -m src.data.ingest
    python -m src.data.ingest --max 500   # quick test run

Output: data/raw/comments_YYYY-MM-DD.jsonl
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.youtube_api import get_service, get_channel_videos, get_video_comments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "channels.yaml"
OUTPUT_DIR = ROOT / "data" / "raw"
CACHE_PATH = ROOT / "data" / "raw" / "seen_ids.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_seen_ids() -> set:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids: set):
    with open(CACHE_PATH, "w") as f:
        json.dump(list(seen_ids), f)


def run(max_comments: int | None = None):
    config = load_config()
    settings = config["settings"]
    channels = config["channels"]
    target = max_comments or settings["target_total"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = OUTPUT_DIR / f"comments_{date_str}.jsonl"

    seen_ids = load_seen_ids()
    service = get_service()

    total_collected = 0
    quota_used = 0

    logger.info("Starting ingestion. Target: %d comments", target)
    logger.info("Output: %s", out_path)

    with open(out_path, "a", encoding="utf-8") as out_file:
        for channel in channels:
            if total_collected >= target:
                break

            channel_id = channel["id"]
            channel_name = channel["name"]
            logger.info("[%s] Fetching videos...", channel_name)

            try:
                videos = get_channel_videos(
                    service,
                    channel_id,
                    max_videos=settings["max_videos_per_channel"],
                )
                # channels.list (1 unit) + playlistItems.list pages (~1 unit each)
                quota_used += 1 + max(1, len(videos) // 50)
            except Exception as e:
                logger.error("[%s] Failed to get videos: %s", channel_name, e)
                continue

            for video in videos:
                if total_collected >= target:
                    break

                video_id = video["video_id"]
                logger.info(
                    "  [%s] video=%s | collected=%d/%d",
                    channel_name, video_id, total_collected, target
                )

                comments_this_video = 0
                try:
                    for comment in get_video_comments(
                        service,
                        video_id,
                        max_comments=settings["max_comments_per_video"],
                        seen_ids=seen_ids,
                    ):
                        # Attach video metadata
                        comment["video_title"] = video["title"]
                        comment["channel_id"] = channel_id
                        comment["channel_name"] = channel_name
                        comment["channel_category"] = channel.get("category", "")

                        out_file.write(json.dumps(comment, ensure_ascii=False) + "\n")
                        total_collected += 1
                        comments_this_video += 1
                        quota_used += 1  # 1 unit per commentThreads.list page

                        if total_collected >= target:
                            break

                except Exception as e:
                    logger.error("  Error fetching comments for %s: %s", video_id, e)
                    continue

                logger.info("    → %d comments collected from this video", comments_this_video)
                time.sleep(0.2)  # be polite to the API

    save_seen_ids(seen_ids)

    logger.info("=" * 50)
    logger.info("Ingestion complete!")
    logger.info("  Total comments: %d", total_collected)
    logger.info("  Estimated quota used: ~%d units", quota_used)
    logger.info("  Output file: %s", out_path)
    logger.info("  Seen IDs cached: %d", len(seen_ids))

    return total_collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max comments to collect")
    args = parser.parse_args()
    run(max_comments=args.max)
