import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, AutoModel
)
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from textblob import TextBlob
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedElectionNLP:
    """
    Advanced NLP analysis for election content including bias detection,
    misinformation risk assessment, and content similarity analysis
    """
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.bias_detector = None
        self.similarity_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
        
        # Bias detection keywords and patterns
        self.bias_indicators = {
            'positive_bias': [
                'amazing', 'incredible', 'outstanding', 'perfect', 'excellent',
                'revolutionary', 'historic', 'unprecedented', 'brilliant', 'genius',
                'hero', 'savior', 'champion', 'visionary', 'remarkable'
            ],
            'negative_bias': [
                'terrible', 'awful', 'disaster', 'corrupt', 'scandal',
                'failure', 'incompetent', 'dangerous', 'radical', 'extreme',
                'destroy', 'devastate', 'chaos', 'crisis', 'threat'
            ],
            'neutral': [
                'reported', 'announced', 'stated', 'according', 'official',
                'confirmed', 'revealed', 'disclosed', 'indicated', 'suggested'
            ]
        }
        
        # Misinformation risk patterns
        self.misinfo_patterns = [
            r'\b(fake news|hoax|conspiracy|cover-up|hidden truth)\b',
            r'\b(they don\'t want you to know|secret agenda|mainstream media lies)\b',
            r'\b(100%|completely|absolutely|never|always)\s+(true|false|wrong|right|proven)\b',
            r'\b(breaking|urgent|exclusive|shocking)\b.*\b(exposed|revealed|uncovered)\b',
            r'\b(wake up|sheep|brainwashed|propaganda)\b',
            r'\b(deep state|elites|globalists)\b',
            r'\b(stolen|rigged|fraud)\b.*\b(election|vote|ballot)\b'
        ]
    
    def _load_models(self):
        """Load and initialize NLP models"""
        try:
            print("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("Loading bias detection model...")
            # Using a more sophisticated approach for bias detection
            self.bias_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            self.bias_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
            self.bias_model.to(self.device)
            
            print("Loading similarity model...")
            self.similarity_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.similarity_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.similarity_model.to(self.device)
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading models: {str(e)}")
            print("Using fallback methods...")
            self.sentiment_analyzer = None
            self.bias_model = None
            self.similarity_model = None
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float, Dict]:
        """
        Advanced sentiment analysis with confidence scores and emotional indicators
        """
        if not text or len(text.strip()) == 0:
            return "neutral", 0.5, {}
        
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            if self.sentiment_analyzer:
                # Use RoBERTa model
                result = self.sentiment_analyzer(cleaned_text[:512])
                label = result[0]['label'].lower()
                score = result[0]['score']
                
                # Map labels to standard format
                label_mapping = {'label_0': 'negative', 'label_1': 'neutral', 'label_2': 'positive'}
                if label in label_mapping:
                    label = label_mapping[label]
                
            else:
                # Fallback to TextBlob
                blob = TextBlob(cleaned_text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label, score = "positive", abs(polarity)
                elif polarity < -0.1:
                    label, score = "negative", abs(polarity)
                else:
                    label, score = "neutral", 1 - abs(polarity)
            
            # Additional emotional indicators
            emotional_indicators = self._analyze_emotional_indicators(cleaned_text)
            
            return label, score, emotional_indicators
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return "neutral", 0.5, {}
    
    def detect_bias(self, text: str, source: str = "") -> Tuple[str, float, Dict]:
        """
        Advanced bias detection using multiple approaches
        """
        if not text or len(text.strip()) == 0:
            return "neutral", 0.5, {}
        
        try:
            cleaned_text = self._clean_text(text)
            bias_scores = {}
            
            # Method 1: Keyword-based bias detection
            keyword_bias = self._keyword_bias_detection(cleaned_text)
            bias_scores['keyword_bias'] = keyword_bias
            
            # Method 2: Sentiment extremity
            sentiment, sent_score, _ = self.analyze_sentiment(cleaned_text)
            extremity_bias = self._sentiment_extremity_bias(sentiment, sent_score)
            bias_scores['sentiment_extremity'] = extremity_bias
            
            # Method 3: Language intensity analysis
            intensity_bias = self._language_intensity_analysis(cleaned_text)
            bias_scores['language_intensity'] = intensity_bias
            
            # Method 4: Source-based bias adjustment
            source_bias = self._source_bias_adjustment(source)
            bias_scores['source_adjustment'] = source_bias
            
            # Method 5: Toxicity-based bias (if model available)
            if self.bias_model:
                toxicity_bias = self._toxicity_bias_detection(cleaned_text)
                bias_scores['toxicity'] = toxicity_bias
            
            # Combine all bias indicators
            overall_bias_score = np.mean(list(bias_scores.values()))
            
            # Determine bias label
            if overall_bias_score > 0.6:
                bias_label = "highly_biased"
            elif overall_bias_score > 0.4:
                bias_label = "moderately_biased"
            elif overall_bias_score > 0.2:
                bias_label = "slightly_biased"
            else:
                bias_label = "neutral"
            
            return bias_label, overall_bias_score, bias_scores
            
        except Exception as e:
            print(f"Error in bias detection: {str(e)}")
            return "neutral", 0.5, {}
    
    def assess_misinformation_risk(self, text: str) -> Tuple[str, float, Dict]:
        """
        Assess misinformation risk using multiple indicators
        """
        if not text or len(text.strip()) == 0:
            return "low_risk", 0.0, {}
        
        try:
            cleaned_text = self._clean_text(text)
            risk_indicators = {}
            
            # Pattern-based detection
            pattern_risk = self._pattern_misinformation_detection(cleaned_text)
            risk_indicators['pattern_based'] = pattern_risk
            
            # Language certainty analysis
            certainty_risk = self._certainty_analysis(cleaned_text)
            risk_indicators['certainty'] = certainty_risk
            
            # Emotional manipulation detection
            emotion_risk = self._emotional_manipulation_detection(cleaned_text)
            risk_indicators['emotional_manipulation'] = emotion_risk
            
            # Source credibility assessment
            credibility_risk = self._source_credibility_risk(text)
            risk_indicators['source_credibility'] = credibility_risk
            
            # Fact-checkable claims detection
            claims_risk = self._fact_checkable_claims_detection(cleaned_text)
            risk_indicators['factual_claims'] = claims_risk
            
            # Calculate overall risk
            risk_weights = {
                'pattern_based': 0.3,
                'certainty': 0.2,
                'emotional_manipulation': 0.2,
                'source_credibility': 0.15,
                'factual_claims': 0.15
            }
            
            overall_risk = sum(
                risk_indicators[key] * risk_weights[key] 
                for key in risk_indicators
            )
            
            # Determine risk level
            if overall_risk > 0.7:
                risk_level = "high_risk"
            elif overall_risk > 0.4:
                risk_level = "medium_risk"
            else:
                risk_level = "low_risk"
            
            return risk_level, overall_risk, risk_indicators
            
        except Exception as e:
            print(f"Error in misinformation assessment: {str(e)}")
            return "low_risk", 0.0, {}
    
    def analyze_content_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Analyze similarity between multiple texts using embeddings
        """
        if not texts or len(texts) < 2:
            return np.array([])
        
        try:
            if self.similarity_model:
                # Use transformer model for embeddings
                embeddings = self._get_embeddings(texts)
                similarity_matrix = cosine_similarity(embeddings)
            else:
                # Fallback to TF-IDF
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Error in similarity analysis: {str(e)}")
            return np.array([])
    
    def cluster_content(self, texts: List[str], n_clusters: int = 5) -> Tuple[List[int], List[str]]:
        """
        Cluster similar content together
        """
        if not texts or len(texts) < n_clusters:
            return [], []
        
        try:
            if self.similarity_model:
                embeddings = self._get_embeddings(texts)
            else:
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                embeddings = vectorizer.fit_transform(texts).toarray()
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Generate cluster themes
            cluster_themes = self._generate_cluster_themes(texts, cluster_labels, n_clusters)
            
            return cluster_labels.tolist(), cluster_themes
            
        except Exception as e:
            print(f"Error in content clustering: {str(e)}")
            return [], []
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text.strip()
    
    def _keyword_bias_detection(self, text: str) -> float:
        """Detect bias using keyword analysis"""
        text_lower = text.lower()
        bias_score = 0
        total_indicators = 0
        
        for category, keywords in self.bias_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category in ['positive_bias', 'negative_bias']:
                        bias_score += 1
                    total_indicators += 1
        
        return min(bias_score / max(total_indicators, 1), 1.0)
    
    def _sentiment_extremity_bias(self, sentiment: str, score: float) -> float:
        """Detect bias based on sentiment extremity"""
        if sentiment in ['positive', 'negative'] and score > 0.8:
            return score
        return 0.0
    
    def _language_intensity_analysis(self, text: str) -> float:
        """Analyze language intensity for bias detection"""
        intensity_indicators = [
            r'\b(very|extremely|incredibly|absolutely|completely|totally)\b',
            r'\b(amazing|terrible|horrible|fantastic|awful|brilliant)\b',
            r'[!]{2,}',  # Multiple exclamation marks
            r'\b[A-Z]{2,}\b',  # ALL CAPS words
        ]
        
        intensity_count = 0
        for pattern in intensity_indicators:
            intensity_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize by text length
        text_length = max(len(text.split()), 1)
        return min(intensity_count / text_length * 10, 1.0)
    
    def _source_bias_adjustment(self, source: str) -> float:
        """Adjust bias score based on known source characteristics"""
        # This is a simplified approach - in practice, you'd have a database
        # of source reliability and bias ratings
        known_biased_sources = ['biased_source1', 'biased_source2']
        known_neutral_sources = ['reuters', 'associated press', 'bbc']
        
        source_lower = source.lower()
        
        if any(bias_source in source_lower for bias_source in known_biased_sources):
            return 0.3
        elif any(neutral_source in source_lower for neutral_source in known_neutral_sources):
            return 0.0
        else:
            return 0.1  # Default slight adjustment for unknown sources
    
    def _toxicity_bias_detection(self, text: str) -> float:
        """Use toxicity model to detect bias"""
        try:
            inputs = self.bias_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bias_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                toxicity_score = probabilities[0][1].item()  # Assuming toxic class is index 1
            
            return toxicity_score
        except:
            return 0.0
    
    def _pattern_misinformation_detection(self, text: str) -> float:
        """Detect misinformation patterns"""
        pattern_count = 0
        for pattern in self.misinfo_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_count += 1
        
        return min(pattern_count / len(self.misinfo_patterns), 1.0)
    
    def _certainty_analysis(self, text: str) -> float:
        """Analyze certainty language that might indicate misinformation"""
        certainty_patterns = [
            r'\b(definitely|certainly|absolutely|guaranteed|100%|never|always)\b',
            r'\b(fact|truth|proven|undeniable|obvious|clear)\b',
            r'\b(everyone knows|nobody can deny|it\'s obvious)\b'
        ]
        
        certainty_count = 0
        for pattern in certainty_patterns:
            certainty_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        text_length = max(len(text.split()), 1)
        return min(certainty_count / text_length * 20, 1.0)
    
    def _emotional_manipulation_detection(self, text: str) -> float:
        """Detect emotional manipulation tactics"""
        manipulation_patterns = [
            r'\b(fear|scared|terrified|panic|disaster|crisis)\b',
            r'\b(outrage|angry|furious|disgusting|shocking)\b',
            r'\b(urgent|immediate|act now|time is running out)\b',
            r'\b(they|them|elites|establishment)\b.*\b(control|manipulate|lie)\b',
            r'\b(wake up|open your eyes|see the truth)\b'
        ]
        
        manipulation_count = 0
        for pattern in manipulation_patterns:
            manipulation_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        text_length = max(len(text.split()), 1)
        return min(manipulation_count / text_length * 15, 1.0)
    
    def _source_credibility_risk(self, text: str) -> float:
        """Assess credibility risk based on text characteristics"""
        credibility_risks = [
            r'\b(unnamed sources?|anonymous sources?|sources? say)\b',
            r'\b(rumor|allegedly|supposedly|claims?)\b',
            r'\b(breaking|exclusive|leaked|insider)\b',
            r'\b(according to|reports suggest|it is believed)\b'
        ]
        
        risk_count = 0
        for pattern in credibility_risks:
            risk_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        text_length = max(len(text.split()), 1)
        return min(risk_count / text_length * 10, 1.0)
    
    def _fact_checkable_claims_detection(self, text: str) -> float:
        """Detect specific factual claims that should be fact-checked"""
        claim_patterns = [
            r'\b\d+%\b',  # Percentages
            r'\b\d+\s+(million|billion|thousand)\b',  # Large numbers
            r'\b(study shows|research proves|statistics show)\b',
            r'\b(according to data|polls show|surveys indicate)\b',
            r'\b(first time|never before|record high|record low)\b'
        ]
        
        claim_count = 0
        for pattern in claim_patterns:
            claim_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        text_length = max(len(text.split()), 1)
        return min(claim_count / text_length * 5, 1.0)
    
    def _analyze_emotional_indicators(self, text: str) -> Dict:
        """Analyze various emotional indicators in text"""
        emotions = {
            'anger': ['angry', 'furious', 'outraged', 'mad', 'livid'],
            'fear': ['afraid', 'scared', 'terrified', 'worried', 'anxious'],
            'joy': ['happy', 'excited', 'thrilled', 'delighted', 'cheerful'],
            'sadness': ['sad', 'depressed', 'disappointed', 'heartbroken'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'appalled']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotions.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = count / len(keywords) if keywords else 0
        
        return emotion_scores
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using transformer model"""
        embeddings = []
        
        for text in texts:
            inputs = self.similarity_tokenizer(
                text[:512], 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.similarity_model(**inputs)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings.append(embedding.cpu().numpy())
        
        return np.array(embeddings)
    
    def _generate_cluster_themes(self, texts: List[str], labels: List[int], n_clusters: int) -> List[str]:
        """Generate themes for each cluster"""
        cluster_themes = []
        
        for cluster_id in range(n_clusters):
            cluster_texts = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
            
            if not cluster_texts:
                cluster_themes.append(f"Empty Cluster {cluster_id}")
                continue
            
            # Use TF-IDF to find most important terms for this cluster
            try:
                vectorizer = TfidfVectorizer(
                    max_features=10, 
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                combined_text = ' '.join(cluster_texts)
                tfidf_matrix = vectorizer.fit_transform([combined_text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Get top terms
                top_indices = scores.argsort()[-3:][::-1]
                top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
                
                if top_terms:
                    theme = ', '.join(top_terms)
                else:
                    theme = f"Cluster {cluster_id}"
                    
                cluster_themes.append(theme)
                
            except:
                cluster_themes.append(f"Cluster {cluster_id}")
        
        return cluster_themes
    
    def analyze_batch(self, df: pd.DataFrame, text_column: str = 'headline', 
                     source_column: str = 'source') -> pd.DataFrame:
        """
        Analyze a batch of texts and return results
        """
        print("🔍 Starting batch analysis...")
        
        results = []
        total_items = len(df)
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Progress: {idx}/{total_items} ({idx/total_items*100:.1f}%)")
            
            text = str(row.get(text_column, ''))
            source = str(row.get(source_column, ''))
            
            # Perform all analyses
            sentiment, sent_score, emotional_indicators = self.analyze_sentiment(text)
            bias_label, bias_score, bias_breakdown = self.detect_bias(text, source)
            misinfo_risk, misinfo_score, misinfo_breakdown = self.assess_misinformation_risk(text)
            
            results.append({
                'index': idx,
                'sentiment': sentiment,
                'sentiment_score': sent_score,
                'bias_label': bias_label,
                'bias_score': bias_score,
                'misinfo_risk': misinfo_risk,
                'misinfo_score': misinfo_score,
                'emotional_anger': emotional_indicators.get('anger', 0),
                'emotional_fear': emotional_indicators.get('fear', 0),
                'emotional_joy': emotional_indicators.get('joy', 0),
                'bias_keyword': bias_breakdown.get('keyword_bias', 0),
                'bias_sentiment_extremity': bias_breakdown.get('sentiment_extremity', 0),
                'bias_language_intensity': bias_breakdown.get('language_intensity', 0),
                'misinfo_patterns': misinfo_breakdown.get('pattern_based', 0),
                'misinfo_certainty': misinfo_breakdown.get('certainty', 0),
                'misinfo_emotional_manipulation': misinfo_breakdown.get('emotional_manipulation', 0)
            })
        
        # Merge results with original dataframe
        results_df = pd.DataFrame(results)
        enhanced_df = df.copy()
        
        for col in results_df.columns:
            if col != 'index':
                enhanced_df[col] = results_df[col].values
        
        print("✅ Batch analysis completed!")
        return enhanced_df
    
    def generate_analysis_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive analysis report
        """
        report = {
            'summary': {},
            'bias_analysis': {},
            'misinformation_analysis': {},
            'sentiment_analysis': {},
            'risk_assessment': {}
        }
        
        # Summary statistics
        report['summary'] = {
            'total_items': len(df),
            'date_range': {
                'start': df['published_at'].min() if 'published_at' in df.columns else 'N/A',
                'end': df['published_at'].max() if 'published_at' in df.columns else 'N/A'
            },
            'sources': df['source'].nunique() if 'source' in df.columns else 0,
            'top_sources': df['source'].value_counts().head().to_dict() if 'source' in df.columns else {}
        }
        
        # Bias analysis
        if 'bias_label' in df.columns:
            report['bias_analysis'] = {
                'bias_distribution': df['bias_label'].value_counts().to_dict(),
                'average_bias_score': df['bias_score'].mean(),
                'highly_biased_percentage': (df['bias_label'] == 'highly_biased').mean() * 100,
                'bias_by_source': df.groupby('source')['bias_score'].mean().to_dict() if 'source' in df.columns else {}
            }
        
        # Misinformation analysis
        if 'misinfo_risk' in df.columns:
            report['misinformation_analysis'] = {
                'risk_distribution': df['misinfo_risk'].value_counts().to_dict(),
                'average_risk_score': df['misinfo_score'].mean(),
                'high_risk_percentage': (df['misinfo_risk'] == 'high_risk').mean() * 100,
                'risk_by_source': df.groupby('source')['misinfo_score'].mean().to_dict() if 'source' in df.columns else {}
            }
        
        # Sentiment analysis
        if 'sentiment' in df.columns:
            report['sentiment_analysis'] = {
                'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
                'average_sentiment_score': df['sentiment_score'].mean(),
                'emotional_indicators': {
                    'anger': df['emotional_anger'].mean() if 'emotional_anger' in df.columns else 0,
                    'fear': df['emotional_fear'].mean() if 'emotional_fear' in df.columns else 0,
                    'joy': df['emotional_joy'].mean() if 'emotional_joy' in df.columns else 0
                }
            }
        
        # Risk assessment
        high_risk_items = 0
        if 'bias_score' in df.columns and 'misinfo_score' in df.columns:
            high_risk_items = len(df[(df['bias_score'] > 0.6) | (df['misinfo_score'] > 0.6)])
        
        report['risk_assessment'] = {
            'high_risk_items': high_risk_items,
            'risk_percentage': (high_risk_items / len(df)) * 100 if len(df) > 0 else 0,
            'recommendations': self._generate_recommendations(df)
        }
        
        return report
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if 'bias_score' in df.columns:
            avg_bias = df['bias_score'].mean()
            if avg_bias > 0.5:
                recommendations.append("High levels of bias detected. Consider implementing bias detection alerts.")
            elif avg_bias > 0.3:
                recommendations.append("Moderate bias levels observed. Monitor content quality closely.")
        
        if 'misinfo_score' in df.columns:
            avg_misinfo = df['misinfo_score'].mean()
            if avg_misinfo > 0.4:
                recommendations.append("Significant misinformation risk detected. Implement fact-checking protocols.")
            elif avg_misinfo > 0.2:
                recommendations.append("Moderate misinformation risk. Consider additional verification steps.")
        
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            if len(source_counts) < 5:
                recommendations.append("Limited source diversity. Consider expanding information sources.")
        
        if len(recommendations) == 0:
            recommendations.append("Content quality appears acceptable. Continue monitoring.")
        
        return recommendations

def main():
    """
    Example usage of the AdvancedElectionNLP class
    """
    print("🚀 Initializing Advanced Election NLP Analyzer...")
    
    # Initialize analyzer
    analyzer = AdvancedElectionNLP()
    
    # Sample data for testing
    sample_texts = [
        "Presidential candidate announces comprehensive healthcare reform plan",
        "BREAKING: Shocking revelation about election fraud discovered!",
        "Polling data shows tight race in key battleground states",
        "This candidate is absolutely the worst choice for our country's future",
        "Fact-checkers verify accuracy of recent policy statements"
    ]
    
    print("\n📝 Analyzing sample texts...")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Analysis {i} ---")
        print(f"Text: {text}")
        
        # Sentiment analysis
        sentiment, sent_score, emotions = analyzer.analyze_sentiment(text)
        print(f"Sentiment: {sentiment} (score: {sent_score:.3f})")
        
        # Bias detection
        bias_label, bias_score, bias_breakdown = analyzer.detect_bias(text)
        print(f"Bias: {bias_label} (score: {bias_score:.3f})")
        
        # Misinformation risk
        misinfo_risk, misinfo_score, misinfo_breakdown = analyzer.assess_misinformation_risk(text)
        print(f"Misinformation Risk: {misinfo_risk} (score: {misinfo_score:.3f})")
    
    # Content similarity analysis
    print(f"\n🔍 Analyzing content similarity...")
    similarity_matrix = analyzer.analyze_content_similarity(sample_texts)
    if similarity_matrix.size > 0:
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Average similarity: {np.mean(similarity_matrix):.3f}")
    
    # Content clustering
    print(f"\n📊 Clustering content...")
    cluster_labels, cluster_themes = analyzer.cluster_content(sample_texts, n_clusters=3)
    if cluster_labels:
        print(f"Cluster labels: {cluster_labels}")
        print(f"Cluster themes: {cluster_themes}")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()