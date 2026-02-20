from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("Youtube_Api_key")

def get_youtube_service():
    return build('youtube', 'v3', developerKey=API_KEY)

# Test: fetch a public channel
youtube = get_youtube_service()
request = youtube.channels().list(
    part='snippet',
    id='UCVHFbw7woebKtFFhjAaDHhQ'  # YouTube Creators channel
)
response = request.execute()

if response.get('items'):
    print("Connected! Channel:", response['items'][0]['snippet']['title'])
else:
    print("Connected but no items returned. API key is working.")
