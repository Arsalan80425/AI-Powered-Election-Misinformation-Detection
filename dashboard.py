import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    st.warning("HuggingFace Transformers not available. Using fallback sentiment analysis.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("TextBlob not available. Using basic sentiment analysis.")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Election Integrity Dashboard", 
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    .author-info {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    .tech-info {
        font-style: italic;
        font-size: 0.8rem;
        opacity: 0.7;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

class ElectionIntegrityAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.bias_detector = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize NLP models with proper error handling"""
        try:
            if HF_AVAILABLE:
                # Try to load HuggingFace model
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                st.success("✅ Advanced AI models loaded successfully!")
            else:
                self.sentiment_analyzer = None
                st.info("ℹ️ Using fallback sentiment analysis methods.")
            
            # Initialize bias detection keywords
            self.bias_detector = {
                'positive_bias': ['amazing', 'excellent', 'outstanding', 'perfect', 'incredible', 
                                'revolutionary', 'historic', 'unprecedented', 'brilliant', 'genius'],
                'negative_bias': ['terrible', 'awful', 'disaster', 'corrupt', 'scandal',
                                'failure', 'incompetent', 'dangerous', 'radical', 'extreme'],
                'neutral': ['reported', 'announced', 'stated', 'according', 'official']
            }
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            self.sentiment_analyzer = None
            self.bias_detector = {
                'positive_bias': ['amazing', 'excellent', 'outstanding', 'perfect', 'incredible'],
                'negative_bias': ['terrible', 'awful', 'disaster', 'corrupt', 'scandal'],
                'neutral': ['reported', 'announced', 'stated', 'according', 'official']
            }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with multiple fallback methods"""
        if not text or len(text.strip()) == 0:
            return "neutral", 0.5
        
        try:
            # Method 1: HuggingFace Transformers (if available)
            if self.sentiment_analyzer is not None:
                result = self.sentiment_analyzer(text[:512])
                scores = {item['label'].lower(): item['score'] for item in result[0]}
                
                best_label = max(scores.keys(), key=lambda x: scores[x])
                best_score = scores[best_label]
                
                # Map RoBERTa labels to standard format
                label_mapping = {
                    'label_0': 'negative', 
                    'label_1': 'neutral', 
                    'label_2': 'positive'
                }
                
                if best_label in label_mapping:
                    best_label = label_mapping[best_label]
                
                return best_label, best_score
            
            # Method 2: TextBlob (if available)
            elif TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    return "positive", abs(polarity)
                elif polarity < -0.1:
                    return "negative", abs(polarity)
                else:
                    return "neutral", 1 - abs(polarity)
            
            # Method 3: Simple keyword-based sentiment (fallback)
            else:
                return self._simple_sentiment_analysis(text)
                    
        except Exception as e:
            # Ultimate fallback
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text):
        """Simple keyword-based sentiment analysis as ultimate fallback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'outstanding', 'brilliant', 'perfect', 'love', 'like', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
                         'corrupt', 'scandal', 'disaster', 'crisis', 'problem', 'issue']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive", min(positive_count / (positive_count + negative_count + 1), 0.9)
        elif negative_count > positive_count:
            return "negative", min(negative_count / (positive_count + negative_count + 1), 0.9)
        else:
            return "neutral", 0.5
    
    def detect_bias(self, text, source):
        """Detect potential bias in text"""
        if not text or len(text.strip()) == 0:
            return "neutral", 0.0
            
        text_lower = text.lower()
        sentiment, confidence = self.analyze_sentiment(text)
        
        # Check for extreme sentiment
        if confidence > 0.8 and sentiment in ['positive', 'negative']:
            return "biased", confidence
        
        # Check for bias keywords
        bias_score = 0
        total_words = len(text_lower.split())
        
        for category, keywords in self.bias_detector.items():
            if category != 'neutral':
                for keyword in keywords:
                    if keyword in text_lower:
                        bias_score += 1
        
        # Check for intensity indicators
        intensity_patterns = [
            r'\b(very|extremely|incredibly|absolutely|completely|totally)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r'\b[A-Z]{2,}\b',  # ALL CAPS words
        ]
        
        for pattern in intensity_patterns:
            matches = len(re.findall(pattern, text))
            bias_score += matches
        
        # Normalize bias score
        normalized_score = min(bias_score / max(total_words * 0.1, 1), 1.0)
        
        if normalized_score > 0.3:
            return "biased", normalized_score
        else:
            return "neutral", 1 - normalized_score
    
    def detect_misinformation_risk(self, text):
        """Detect potential misinformation indicators"""
        if not text or len(text.strip()) == 0:
            return "low_risk"
            
        risk_indicators = [
            r'\b(fake|hoax|conspiracy|cover-up)\b',
            r'\b(they don\'t want you to know|hidden truth|secret)\b',
            r'\b(100%|completely|absolutely|never|always)\b.*\b(true|false|wrong|right)\b',
            r'\b(breaking|urgent|exclusive)\b.*\b(scandal|exposed|revealed)\b',
            r'\b(wake up|sheep|brainwashed)\b',
            r'\b(stolen|rigged|fraud)\b.*\b(election|vote)\b'
        ]
        
        risk_score = 0
        for pattern in risk_indicators:
            if re.search(pattern, text.lower()):
                risk_score += 1
        
        # Normalize risk score
        risk_level = min(risk_score / len(risk_indicators), 1.0)
        
        if risk_level > 0.6:
            return "high_risk"
        elif risk_level > 0.3:
            return "medium_risk"
        else:
            return "low_risk"

# Initialize analyzer with caching
@st.cache_resource
def get_analyzer():
    return ElectionIntegrityAnalyzer()

@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample election data with caching"""
    np.random.seed(42)
    
    sources = ['CNN', 'Fox News', 'BBC', 'Reuters', 'Associated Press', 
               'The Guardian', 'Wall Street Journal', 'Twitter User', 'Facebook Post',
               'Politico', 'The Hill', 'NPR']
    
    # More realistic headline templates
    headline_templates = [
        "Presidential candidate announces new {policy} policy",
        "Voter turnout reaches record levels in {state}",
        "Election officials report {status} in voting process",
        "Candidate's {event} draws {number} supporters",
        "Latest poll shows {candidate} leading in {state}",
        "Supreme Court ruling affects {topic} voting rights",
        "Social media platforms address election misinformation",
        "International observers praise election transparency",
        "Debate focuses on {policy} and economic recovery",
        "Early voting shows strong participation nationwide",
        "BREAKING: {candidate} makes controversial statement about {topic}",
        "EXCLUSIVE: Investigation reveals {topic} irregularities",
        "Fact-checkers debunk false claims about {topic}",
        "Swing state {state} becomes election battleground"
    ]
    
    # Data for template filling
    policies = ['healthcare', 'education', 'climate', 'immigration', 'economic', 'tax']
    states = ['Pennsylvania', 'Wisconsin', 'Michigan', 'Arizona', 'Georgia', 'Nevada']
    candidates = ['Johnson', 'Smith', 'Davis', 'Wilson', 'Brown']
    topics = ['voting machines', 'mail-in ballots', 'voter registration', 'poll workers']
    events = ['rally', 'town hall', 'fundraiser', 'press conference']
    numbers = ['thousands of', 'hundreds of', 'record numbers of']
    statuses = ['smooth operations', 'minor delays', 'high efficiency', 'technical issues']
    
    data = []
    
    # Pre-generate all data without analysis first (much faster)
    raw_data = []
    for i in range(800):  # Reduced from 1200 to 800 for better performance
        template = np.random.choice(headline_templates)
        headline = template.format(
            policy=np.random.choice(policies),
            state=np.random.choice(states),
            candidate=np.random.choice(candidates),
            topic=np.random.choice(topics),
            event=np.random.choice(events),
            number=np.random.choice(numbers),
            status=np.random.choice(statuses)
        )
        
        source = np.random.choice(sources)
        days_ago = np.random.randint(0, 30)
        hours_ago = np.random.randint(0, 24)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
        
        raw_data.append({
            'headline': headline,
            'source': source,
            'timestamp': timestamp,
            'engagement': np.random.randint(10, 10000),
            'shares': np.random.randint(1, 1000)
        })
    
    # Now analyze in batches (more efficient)
    analyzer = get_analyzer()
    
    for item in raw_data:
        # Fast analysis
        sentiment, sentiment_score = analyzer.analyze_sentiment(item['headline'])
        bias_label, bias_score = analyzer.detect_bias(item['headline'], item['source'])
        misinfo_risk = analyzer.detect_misinformation_risk(item['headline'])
        
        data.append({
            **item,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'bias_label': bias_label,
            'bias_score': bias_score,
            'misinfo_risk': misinfo_risk
        })
    
    return pd.DataFrame(data)

def main():
    # Header with similar styling to your health project
    st.markdown("""
    <div class="header-container">
        <div class="main-title">🗳️ AI-Powered Election Misinformation Detection</div>
        <div class="subtitle">Advanced Democracy Protection System</div>
        <div class="author-info">🎯 Built for Democratic Integrity Initiative by Mohammed Arsalan</div>
        <div class="tech-info">Leveraging RoBERTa, NLP Pipelines & Real-time Content Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check system status
    with st.expander("System Status", expanded=False):
        st.write("**Component Status:**")
        st.write(f"- HuggingFace Transformers: {'✅ Available' if HF_AVAILABLE else '❌ Not Available'}")
        st.write(f"- TextBlob: {'✅ Available' if TEXTBLOB_AVAILABLE else '❌ Not Available'}")
        st.write(f"- WordCloud: {'✅ Available' if WORDCLOUD_AVAILABLE else '❌ Not Available'}")
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Load data only once with caching
    if 'data_loaded' not in st.session_state:
        with st.spinner("🔄 Loading election data (this happens only once)..."):
            st.session_state.df = generate_sample_data()
            st.session_state.data_loaded = True
        st.sidebar.success(f"✅ Loaded {len(st.session_state.df)} articles")
    else:
        st.sidebar.info(f"📊 {len(st.session_state.df)} articles ready")
    
    df = st.session_state.df
    
    # Date filter
    if not df.empty:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
        
        # Source filter
        sources = st.sidebar.multiselect(
            "Select News Sources",
            options=df['source'].unique(),
            default=df['source'].unique()
        )
        
        # Filter data
        if len(date_range) == 2:
            filtered_df = df[
                (df['timestamp'].dt.date >= date_range[0]) & 
                (df['timestamp'].dt.date <= date_range[1]) &
                (df['source'].isin(sources))
            ]
        else:
            filtered_df = df[df['source'].isin(sources)]
    else:
        st.error("No data available")
        return
    
    # Key Metrics
    st.header("📊 Key Metrics")
    
    if not filtered_df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Articles", len(filtered_df))
        
        with col2:
            biased_count = len(filtered_df[filtered_df['bias_label'] == 'biased'])
            st.metric("Biased Content", biased_count, f"{biased_count/len(filtered_df)*100:.1f}%")
        
        with col3:
            high_risk = len(filtered_df[filtered_df['misinfo_risk'] == 'high_risk'])
            st.metric("High Misinfo Risk", high_risk, f"{high_risk/len(filtered_df)*100:.1f}%")
        
        with col4:
            avg_engagement = int(filtered_df['engagement'].mean())
            st.metric("Avg Engagement", f"{avg_engagement:,}")
        
        with col5:
            sources_count = filtered_df['source'].nunique()
            st.metric("Active Sources", sources_count)
        
        # Main Dashboard
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Bias Analysis", "⚠️ Misinformation Trends", "📈 Sentiment Timeline", "🔗 Network Analysis"])
        
        with tab1:
            st.header("Bias Detection Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bias by source
                bias_by_source = filtered_df.groupby(['source', 'bias_label']).size().unstack(fill_value=0)
                
                if not bias_by_source.empty:
                    bias_reset = bias_by_source.reset_index()
                    bias_columns = [col for col in bias_reset.columns if col != 'source']
                    
                    bias_melted = pd.melt(
                        bias_reset, 
                        id_vars=['source'], 
                        value_vars=bias_columns,
                        var_name='bias_type', 
                        value_name='count'
                    )
                    
                    fig_bias = px.bar(
                        bias_melted,
                        x='source',
                        y='count',
                        color='bias_type',
                        title="Bias Distribution by News Source",
                        color_discrete_map={'biased': '#ff6b6b', 'neutral': '#4ecdc4'}
                    )
                    fig_bias.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_bias, use_container_width=True)
                else:
                    st.info("No bias data available for the selected filters.")
            
            with col2:
                # Bias score distribution
                if not filtered_df.empty:
                    fig_score = px.histogram(
                        filtered_df,
                        x='bias_score',
                        color='bias_label',
                        title="Bias Score Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
                else:
                    st.info("No data available for bias score distribution.")
        
        with tab2:
            st.header("Misinformation Risk Analysis")
            
            # Risk over time
            if not filtered_df.empty:
                daily_risk = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'misinfo_risk']).size().unstack(fill_value=0)
                
                if not daily_risk.empty:
                    daily_risk_reset = daily_risk.reset_index()
                    risk_columns = [col for col in daily_risk_reset.columns if col != 'timestamp']
                    
                    daily_risk_melted = pd.melt(
                        daily_risk_reset, 
                        id_vars=['timestamp'], 
                        value_vars=risk_columns,
                        var_name='risk_level', 
                        value_name='count'
                    )
                    
                    fig_risk_time = px.area(
                        daily_risk_melted,
                        x='timestamp',
                        y='count',
                        color='risk_level',
                        title="Misinformation Risk Trends Over Time",
                        color_discrete_map={'high_risk': '#ff4757', 'medium_risk': '#ffa502', 'low_risk': '#2ed573'}
                    )
                    st.plotly_chart(fig_risk_time, use_container_width=True)
                else:
                    st.info("No risk trend data available.")
            else:
                st.info("No data available for risk analysis.")
        
        with tab3:
            st.header("Sentiment Analysis Timeline")
            
            # Sentiment over time
            if not filtered_df.empty:
                daily_sentiment = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                
                if not daily_sentiment.empty:
                    daily_sentiment_reset = daily_sentiment.reset_index()
                    sentiment_columns = [col for col in daily_sentiment_reset.columns if col != 'timestamp']
                    
                    daily_sentiment_melted = pd.melt(
                        daily_sentiment_reset, 
                        id_vars=['timestamp'], 
                        value_vars=sentiment_columns,
                        var_name='sentiment_type', 
                        value_name='count'
                    )
                    
                    fig_sentiment = px.line(
                        daily_sentiment_melted,
                        x='timestamp',
                        y='count',
                        color='sentiment_type',
                        title="Sentiment Trends Over Time",
                        color_discrete_map={'positive': '#2ed573', 'negative': '#ff4757', 'neutral': '#57606f'}
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                else:
                    st.info("No sentiment trend data available.")
            else:
                st.info("No data available for sentiment analysis.")
        
        with tab4:
            st.header("Information Flow Network Analysis")
            
            # Create simple network visualization
            if not filtered_df.empty:
                G = nx.Graph()
                
                # Add nodes (sources)
                for source in filtered_df['source'].unique():
                    G.add_node(source)
                
                # Add edges based on sentiment similarity
                source_sentiment = filtered_df.groupby('source')['sentiment'].apply(list).to_dict()
                
                for source1 in source_sentiment:
                    for source2 in source_sentiment:
                        if source1 != source2:
                            common_sentiment = len(set(source_sentiment[source1]) & set(source_sentiment[source2]))
                            if common_sentiment > 1:
                                G.add_edge(source1, source2, weight=common_sentiment)
                
                # Network statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", len(G.nodes()))
                with col2:
                    st.metric("Total Connections", len(G.edges()))
                with col3:
                    density = nx.density(G) if len(G.nodes()) > 0 else 0
                    st.metric("Network Density", f"{density:.3f}")
            else:
                st.info("No data available for network analysis.")

        # Recent Alerts Section (moved before tabs)
        st.header("🚨 Recent Content Alerts")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Potentially Biased Content")
            biased_df = filtered_df[filtered_df['bias_label'] == 'biased'].sort_values('timestamp', ascending=False)
    
            if not biased_df.empty:
                for idx, row in biased_df.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div style="border-left: 4px solid #ff6b6b; padding: 1rem; margin: 0.5rem 0; background-color: #2d3748; border-radius: 5px; border: 1px solid #4a5568;">
                            <h4 style="margin: 0; color: #ff6b6b;">{row['source']}</h4>
                            <p style="margin: 0.5rem 0; font-weight: bold; color: #e2e8f0;">{row['headline']}</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #a0aec0;">
                                <span>📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
                                <span>🎯 Bias Score: {row['bias_score']:.2f}</span>
                                <span>💬 {row['engagement']:,} engagements</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("No biased content detected in the selected period.")

        with col2:
            st.subheader("High Misinformation Risk")
            high_risk_df = filtered_df[filtered_df['misinfo_risk'] == 'high_risk'].sort_values('engagement', ascending=False)
    
            if not high_risk_df.empty:
                for idx, row in high_risk_df.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div style="border-left: 4px solid #ff4757; padding: 1rem; margin: 0.5rem 0; background-color: #2d3748; border-radius: 5px; border: 1px solid #4a5568;">
                            <h4 style="margin: 0; color: #ff4757;">{row['source']}</h4>
                            <p style="margin: 0.5rem 0; font-weight: bold; color: #e2e8f0;">{row['headline']}</p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #a0aec0;">
                                <span>📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
                                <span>⚠️ High Risk</span>
                                <span>💬 {row['engagement']:,} engagements</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("No high-risk misinformation detected.")
        
        
    
    # Data Export
    st.sidebar.header("Data Export")
    if st.sidebar.button("Download Analysis Report"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"election_integrity_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    # Footer
    st.markdown("---")
    st.markdown("**Election Integrity Dashboard** - AI-Powered Democracy Protection")

if __name__ == "__main__":
    main()