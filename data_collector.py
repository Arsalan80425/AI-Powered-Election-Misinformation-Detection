import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
from typing import List, Dict, Optional
import csv
import re

class ElectionDataCollector:
    """
    Collects election-related data from various sources for analysis
    """
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def collect_news_data(self, query: str = "election", 
                         days_back: int = 30,
                         sources: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect news articles from NewsAPI
        """
        if not self.news_api_key:
            print("Warning: NEWS_API_KEY not found. Using sample data.")
            return self._generate_sample_news_data(query, days_back)
        
        # Define election-related keywords
        election_keywords = [
            "election", "voting", "ballot", "candidate", "campaign",
            "democracy", "poll", "voter", "electoral", "political"
        ]
        
        all_articles = []
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        for keyword in election_keywords[:3]:  # Limit API calls
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'from': from_date.strftime('%Y-%m-%d'),
                    'to': to_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'pageSize': 100
                }
                
                if sources:
                    params['sources'] = ','.join(sources)
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        all_articles.append({
                            'headline': article.get('title', ''),
                            'content': article.get('description', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'keyword': keyword
                        })
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error collecting data for keyword '{keyword}': {str(e)}")
                continue
        
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Clean and process data
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.drop_duplicates(subset=['headline', 'source'])
            df = df.dropna(subset=['headline'])
        
        return df
    
    def _generate_sample_news_data(self, query: str, days_back: int) -> pd.DataFrame:
        """
        Generate realistic sample election data for demonstration
        """
        np.random.seed(42)
        
        # Realistic news sources
        sources = [
            'CNN', 'Fox News', 'BBC News', 'Reuters', 'Associated Press',
            'The Guardian', 'Wall Street Journal', 'New York Times',
            'Washington Post', 'NPR', 'Politico', 'The Hill'
        ]
        
        # Realistic election headlines templates
        headline_templates = [
            "Presidential candidate {name} proposes new {policy} plan",
            "{name} leads in latest {state} polling data",
            "Voter turnout expected to reach record highs in {state}",
            "Election officials report {issue} in {state} voting",
            "{name}'s campaign rally draws thousands in {city}",
            "Supreme Court decision impacts {issue} voting rights",
            "Social media platforms combat election misinformation about {topic}",
            "International observers praise {country}'s election transparency",
            "Debate between candidates focuses on {policy} and economy",
            "Early voting numbers show strong participation in {state}",
            "Election security measures upgraded in battleground states",
            "{name} announces running mate selection for upcoming election",
            "Polling stations report smooth operations despite high turnout",
            "Fact-checkers identify misleading claims about {topic}",
            "Voter registration drives target young demographics in {state}",
            "Campaign finance reports reveal major donor contributions",
            "Election officials debunk false claims about voting machines",
            "Swing state {state} becomes key battleground for candidates",
            "Mail-in ballot procedures face legal challenges in {state}",
            "Exit polls suggest close race in critical districts"
        ]
        
        # Sample data for templates
        candidate_names = ['Johnson', 'Smith', 'Davis', 'Wilson', 'Brown', 'Taylor']
        policies = ['healthcare', 'education', 'immigration', 'climate', 'economic', 'tax reform']
        states = ['Pennsylvania', 'Wisconsin', 'Michigan', 'Arizona', 'Georgia', 'Nevada', 'Florida']
        cities = ['Philadelphia', 'Milwaukee', 'Detroit', 'Phoenix', 'Atlanta', 'Las Vegas']
        issues = ['technical glitches', 'long lines', 'smooth processing', 'record turnout']
        topics = ['vote counting', 'mail-in ballots', 'voter eligibility', 'polling locations']
        
        data = []
        for i in range(1500):  # Generate more comprehensive dataset
            template = np.random.choice(headline_templates)
            source = np.random.choice(sources)
            
            # Fill template with random data
            headline = template.format(
                name=np.random.choice(candidate_names),
                policy=np.random.choice(policies),
                state=np.random.choice(states),
                city=np.random.choice(cities),
                issue=np.random.choice(issues),
                topic=np.random.choice(topics),
                country="United States"
            )
            
            # Add some bias-inducing prefixes occasionally
            bias_prefixes = ['BREAKING:', 'EXCLUSIVE:', 'SHOCKING:', 'URGENT:', '']
            prefix = np.random.choice(bias_prefixes, p=[0.1, 0.1, 0.05, 0.05, 0.7])
            if prefix:
                headline = f"{prefix} {headline}"
            
            # Generate realistic content
            content_templates = [
                f"According to recent reports, {headline.lower()}. This development has significant implications for the upcoming election.",
                f"Sources confirm that {headline.lower()}. Political analysts are closely monitoring the situation.",
                f"In a recent statement, officials announced that {headline.lower()}. The public response has been mixed.",
                f"Breaking developments show that {headline.lower()}. This could impact voter sentiment significantly."
            ]
            content = np.random.choice(content_templates)
            
            # Generate timestamp within the specified range
            days_ago = np.random.randint(0, days_back)
            hours_ago = np.random.randint(0, 24)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            data.append({
                'headline': headline,
                'content': content,
                'source': source,
                'url': f"https://example-news.com/article/{i}",
                'published_at': timestamp,
                'keyword': query
            })
        
        return pd.DataFrame(data)
    
    def collect_reddit_data(self, subreddits: List[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Collect election-related posts from Reddit (using public API)
        """
        if subreddits is None:
            subreddits = ['politics', 'PoliticalDiscussion', 'moderatepolitics', 'Ask_Politics']
        
        all_posts = []
        
        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                params = {'limit': limit}
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    for post in posts:
                        post_data = post.get('data', {})
                        
                        # Filter for election-related content
                        title = post_data.get('title', '').lower()
                        election_keywords = ['election', 'vote', 'candidate', 'campaign', 'ballot', 'poll']
                        
                        if any(keyword in title for keyword in election_keywords):
                            all_posts.append({
                                'headline': post_data.get('title', ''),
                                'content': post_data.get('selftext', ''),
                                'source': f"r/{subreddit}",
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'published_at': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                'score': post_data.get('score', 0),
                                'num_comments': post_data.get('num_comments', 0)
                            })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error collecting Reddit data from r/{subreddit}: {str(e)}")
                continue
        
        return pd.DataFrame(all_posts)
    
    def collect_twitter_sample_data(self) -> pd.DataFrame:
        """
        Generate sample Twitter-like election data (since Twitter API requires authentication)
        """
        np.random.seed(42)
        
        twitter_templates = [
            "Just voted! Make sure you get out there too! #{hashtag} #Vote2024",
            "Breaking: New poll shows tight race in {state} #{hashtag}",
            "Can't believe what {candidate} just said about {topic}... #{hashtag}",
            "Fact check: Claims about {topic} are misleading #{hashtag} #FactCheck",
            "Long lines at my polling station but democracy is worth the wait! #Vote2024",
            "This election will determine the future of {topic} #{hashtag}",
            "THREAD: Why you should care about {topic} in this election 1/",
            "Misinformation alert: False claims circulating about {topic} #{hashtag}",
            "Record turnout reported in {state}! Democracy in action #{hashtag}",
            "Remember: Your vote is your voice! #ElectionDay #{hashtag}"
        ]
        
        hashtags = ['Politics', 'Election2024', 'Democracy', 'VoteBlue', 'VoteRed', 'IndependentVoter']
        candidates = ['Johnson', 'Smith', 'Davis']
        states = ['PA', 'WI', 'MI', 'AZ', 'GA', 'NV']
        topics = ['healthcare', 'economy', 'education', 'climate', 'immigration']
        
        data = []
        for i in range(800):
            template = np.random.choice(twitter_templates)
            
            tweet = template.format(
                hashtag=np.random.choice(hashtags),
                state=np.random.choice(states),
                candidate=np.random.choice(candidates),
                topic=np.random.choice(topics)
            )
            
            # Generate engagement metrics
            retweets = np.random.randint(0, 1000)
            likes = np.random.randint(retweets, retweets * 3 + 100)
            replies = np.random.randint(0, retweets // 2 + 10)
            
            # Generate timestamp
            days_ago = np.random.randint(0, 30)
            hours_ago = np.random.randint(0, 24)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            data.append({
                'headline': tweet,
                'content': tweet,  # For tweets, headline and content are the same
                'source': 'Twitter',
                'url': f"https://twitter.com/user{i}/status/{1000000 + i}",
                'published_at': timestamp,
                'retweets': retweets,
                'likes': likes,
                'replies': replies,
                'engagement_total': retweets + likes + replies
            })
        
        return pd.DataFrame(data)
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Save collected data to file
        """
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        elif format.lower() == 'json':
            df.to_json(filename, orient='records', date_format='iso')
            print(f"Data saved to {filename}")
        else:
            raise ValueError("Format must be 'csv' or 'json'")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from file
        """
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.endswith('.json'):
            df = pd.read_json(filename)
        else:
            raise ValueError("File must be .csv or .json")
        
        # Ensure published_at is datetime
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
        
        return df
    
    def combine_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple datasets into one
        """
        # Standardize column names across datasets
        standardized_datasets = []
        
        for df in datasets:
            df_copy = df.copy()
            
            # Ensure all datasets have required columns
            required_columns = ['headline', 'content', 'source', 'published_at']
            for col in required_columns:
                if col not in df_copy.columns:
                    if col == 'content':
                        df_copy[col] = df_copy['headline']  # Use headline as content if missing
                    else:
                        df_copy[col] = 'Unknown'
            
            standardized_datasets.append(df_copy)
        
        # Combine all datasets
        combined_df = pd.concat(standardized_datasets, ignore_index=True)
        
        # Remove duplicates based on headline and source
        combined_df = combined_df.drop_duplicates(subset=['headline', 'source'])
        
        # Sort by publication date
        combined_df = combined_df.sort_values('published_at', ascending=False)
        
        return combined_df

def main():
    """
    Main function to collect and save election data
    """
    collector = ElectionDataCollector()
    
    print("🗳️ Starting Election Data Collection...")
    
    # Collect data from different sources
    print("\n📰 Collecting news data...")
    news_df = collector.collect_news_data(days_back=30)
    print(f"Collected {len(news_df)} news articles")
    
    print("\n🐦 Generating Twitter sample data...")
    twitter_df = collector.collect_twitter_sample_data()
    print(f"Generated {len(twitter_df)} Twitter posts")
    
    print("\n📱 Collecting Reddit data...")
    reddit_df = collector.collect_reddit_data()
    print(f"Collected {len(reddit_df)} Reddit posts")
    
    # Combine all datasets
    print("\n🔄 Combining datasets...")
    all_datasets = [df for df in [news_df, twitter_df, reddit_df] if not df.empty]
    
    if all_datasets:
        combined_df = collector.combine_datasets(all_datasets)
        print(f"Combined dataset contains {len(combined_df)} total items")
        
        # Save the combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"election_data_{timestamp}.csv"
        collector.save_data(combined_df, filename)
        
        # Display summary statistics
        print(f"\n📊 Dataset Summary:")
        print(f"Total items: {len(combined_df)}")
        print(f"Date range: {combined_df['published_at'].min()} to {combined_df['published_at'].max()}")
        print(f"Sources: {combined_df['source'].nunique()}")
        print(f"Top sources: {combined_df['source'].value_counts().head().to_dict()}")
        
        return combined_df
    else:
        print("❌ No data collected from any source")
        return pd.DataFrame()

if __name__ == "__main__":
    collected_data = main()