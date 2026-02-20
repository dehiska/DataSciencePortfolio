"""
YouTube Data API v3 wrapper.
Uses API Key (no OAuth) — sufficient for all public comment data.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Iterator

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("Youtube_Api_key")

# Quota cost reference (units per call)
# videos.list          → 1 unit
# commentThreads.list  → 1 unit (returns up to 100 comments)
# search.list          → 100 units  ← avoid, expensive


def get_service():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def get_channel_videos(service, channel_id: str, max_videos: int = 10) -> list[dict]:
    """
    Return a list of recent video dicts {video_id, title, published_at}
    for the given channel.

    Uses channels.list (1 unit) + playlistItems.list (1 unit/page, 50 videos/page)
    instead of search.list (100 units/call) — 50-100x cheaper on quota.
    """
    # Step 1: Get the channel's "uploads" playlist ID (1 unit)
    try:
        ch_resp = service.channels().list(
            part="contentDetails",
            id=channel_id,
        ).execute()
    except HttpError as e:
        logger.error("channels.list error for %s: %s", channel_id, e)
        return []

    items = ch_resp.get("items", [])
    if not items:
        logger.warning("Channel not found or no contentDetails: %s", channel_id)
        return []

    uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Step 2: Page through the uploads playlist (1 unit/page, 50 videos/page)
    videos = []
    next_page = None

    while len(videos) < max_videos:
        try:
            resp = service.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist,
                maxResults=min(50, max_videos - len(videos)),
                pageToken=next_page,
            ).execute()
        except HttpError as e:
            logger.error("playlistItems.list error for %s: %s", channel_id, e)
            break

        for item in resp.get("items", []):
            snippet = item["snippet"]
            videos.append({
                "video_id": snippet["resourceId"]["videoId"],
                "title": snippet["title"],
                "published_at": snippet["publishedAt"],
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle", ""),
            })

        next_page = resp.get("nextPageToken")
        if not next_page or len(videos) >= max_videos:
            break
        time.sleep(0.1)

    return videos[:max_videos]


def get_video_comments(
    service,
    video_id: str,
    max_comments: int = 100,
    seen_ids: set | None = None,
) -> Iterator[dict]:
    """
    Yield comment dicts for a single video.
    Each commentThreads.list call costs 1 quota unit and returns up to 100 comments.
    Includes top-level comments and their replies.
    """
    if seen_ids is None:
        seen_ids = set()

    next_page = None
    fetched = 0

    while fetched < max_comments:
        try:
            resp = service.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(100, max_comments - fetched),
                pageToken=next_page,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            # Comments disabled or video not found — skip quietly
            if e.resp.status in (403, 404):
                logger.debug("Comments unavailable for video %s: %s", video_id, e)
            else:
                logger.error("Comment fetch error for video %s: %s", video_id, e)
            break

        for thread in resp.get("items", []):
            top = thread["snippet"]["topLevelComment"]["snippet"]
            comment_id = thread["snippet"]["topLevelComment"]["id"]

            if comment_id in seen_ids:
                continue
            seen_ids.add(comment_id)

            yield _build_comment_record(
                comment_id=comment_id,
                text=top["textDisplay"],
                author_hash=_hash_author(top.get("authorDisplayName", "")),
                like_count=top.get("likeCount", 0),
                published_at=top["publishedAt"],
                video_id=video_id,
                parent_id=None,
                reply_count=thread["snippet"].get("totalReplyCount", 0),
            )
            fetched += 1

            # Include replies if present
            for reply in thread.get("replies", {}).get("comments", []):
                r = reply["snippet"]
                rid = reply["id"]
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                yield _build_comment_record(
                    comment_id=rid,
                    text=r["textDisplay"],
                    author_hash=_hash_author(r.get("authorDisplayName", "")),
                    like_count=r.get("likeCount", 0),
                    published_at=r["publishedAt"],
                    video_id=video_id,
                    parent_id=comment_id,
                    reply_count=0,
                )
                fetched += 1

        next_page = resp.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.1)


def _build_comment_record(
    comment_id, text, author_hash, like_count,
    published_at, video_id, parent_id, reply_count
) -> dict:
    return {
        "content_id": f"youtube:{comment_id}",
        "platform": "youtube",
        "comment_id": comment_id,
        "video_id": video_id,
        "parent_id": parent_id,
        "text_raw": text,
        "like_count": like_count,
        "reply_count": reply_count,
        "author_hash": author_hash,
        "published_at": published_at,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        # Labels filled in later by model
        "label_toxicity": None,
        "label_hate_racism": None,
        "label_harassment": None,
        "model_version": None,
    }


def _hash_author(name: str) -> str:
    """One-way hash author name — never store raw usernames (per spec)."""
    import hashlib
    return hashlib.sha256(name.encode()).hexdigest()[:16] if name else ""
