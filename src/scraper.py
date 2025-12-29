import numpy as np
import pandas as pd
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from googleapiclient.discovery import build
import time
import praw

class BaseScraper():
    def __init__(self, query, max_results=100):
        self.query = query
        self.max_results = max_results

    def fetch_data(self):
        raise NotImplementedError("Subclasses must implement this method")

class RedditScraper(BaseScraper):
    def __init__(self, query, client_id, client_secret, user_agent, subreddit, max_results):
        super().__init__(query, max_results)
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit = subreddit if subreddit else ['productreview']

    def _fetch_from_subreddit(self, reddit, subreddit_name):
        sub = reddit.subreddit(subreddit_name)
        posts = sub.search(self.query, limit=self.max_results)
    
        # Parsing the posts
        parsed_posts = []
        for post in posts:
            parsed_posts.append({
                "title": post.title,
                "review": post.selftext,
                "source": f"Reddit r/ {subreddit_name}",
                "date": datetime.fromtimestamp(post.created_utc),
                "url": f"https://reddit.com{post.permalink}",
                "upvotes": post.score,
                "comments": post.num_comments
            })
            
        df = pd.DataFrame(parsed_posts)
        
        return df
        
    def fetch_data(self):
        """
        Fetches reviews from a subreddit based on a query.
        
        Args:
            query (str): Keyword to search for (e.g., "iPhone").
            subreddit (str): Subreddit to search in (default: "productreview").
            limit (int): Maximum number of posts to fetch (default: 1000).
        
        Returns:
            List of dicts with post details.
        """
        # Initializing Reddit API client
        reddit = praw.Reddit(client_id=self.client_id, client_secret=self.client_secret, user_agent=self.user_agent)
        full_df = []
        
        for subreddit_name in self.subreddit:
            try:
                print(f"Fetching from r/{subreddit_name}")
                subreddit_name_df = self._fetch_from_subreddit(reddit, subreddit_name)
                #sorted_df = df.sort_values(by='upvotes', ascending=False).reset_index(drop=True)
                #slice_df = sorted_df[:int(percent*len(sorted_df))]
                full_df.append(subreddit_name_df)
            except Exception as e:
                print(f"Error fetching r/{subreddit_name}: {e}")
         
        if not full_df:
            return pd.DataFrame()

        all_df = pd.concat(full_df, ignore_index=True).drop_duplicates()
        return all_df

class YoutubeScraper(BaseScraper):
    def __init__(self, query, youtube_api_key, max_comments, sort_by, max_videos=10, top_n=5):
        super().__init__(query, max_videos)
        self.query = query
        self.youtube_api_key = youtube_api_key
        self.max_comments = max_comments
        self.top_n = top_n
        self.sort_by = sort_by
        self.youtube = self._get_youtube_service()

    def _get_youtube_service(self):
        return build('youtube', 'v3', developerKey=self.youtube_api_key)
    
    def _search_youtube_videos(self):
        # Step 2: Search for videos
        search_response = self.youtube.search().list(
            q=self.query,
            part="id",
            type="video",
            maxResults=self.max_results # because the Base attribute is set to max_results for max_videos object
        ).execute()
    
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        if not video_ids:
            return pd.DataFrame()  # No results
        
        # Step 3: Get video stats
        videos_response = self.youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        ).execute()
    
        # Step 4: Collect and sort by views
        video_data = []
        for item in videos_response['items']:
            stats = item['statistics']
            snippet = item['snippet']
            metric = int(stats.get(self.sort_by, 0)) if self.sort_by in stats else 0
            
            video_data.append({
                "video_id": item["id"],
                "title": snippet["title"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentsCount", 0)),
                "published_at": snippet["publishedAt"],
                "sort_metric": metric
            })
    
        # Sort by view count and return top_n
        sorted_videos = sorted(video_data, key=lambda x: x["sort_metric"], reverse=True)
        video_details = [{'video_id':video['video_id'],'title':video['title']} for video in sorted_videos[:self.top_n]]
        
        return video_details    
        
    def _get_video_comments(self, vid, title):
        comments = []
        next_page_token = None
    
        while len(comments) < self.max_comments:
            response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=vid,
                maxResults=min(100, self.max_comments - len(comments)),
                textFormat='plainText',
                pageToken=next_page_token
            ).execute()
    
            for item in response['items']:
                comment_data = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'Video_ID': vid,
                    'Video_title': title,
                    'Author': comment_data['authorDisplayName'],
                    'Text': comment_data['textDisplay'],
                    'PublishedAt': comment_data['publishedAt'],
                    'LikeCount': comment_data['likeCount']
                })
    
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    
            time.sleep(0.2)  # avoid hitting API quota too fast
    
        return comments
    

    def fetch_data(self):
        video_details = self._search_youtube_videos()

        all_comments = []
        for video_detail in video_details:
            vid = video_detail['video_id']
            title = video_detail['title']
            print(f"Fetching comments for video: {vid} Title: {title}")
            all_comments.extend(self._get_video_comments(vid, title))

        return pd.DataFrame(all_comments)