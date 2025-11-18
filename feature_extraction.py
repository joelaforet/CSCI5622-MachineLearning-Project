"""
Interview Communication Feature Extraction Pipeline
===================================================
This module extracts linguistic features from job interview transcripts to classify
the degree of explanation in responses (under-explained, succinct, comprehensive, over-explained).

Author: ML Project Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Syntactic Features
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk import pos_tag, word_tokenize

# Semantic Features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation

# Advanced Features (conditional import)
try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Note: PyTorch/Transformers not available. BERT features will be disabled.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')


class InterviewDataLoader:
    """
    Loads and preprocesses interview transcript data from the VetTrain dataset.
    Combines question-answer pairs into single samples with their labels.
    """
    
    def __init__(self, transcript_dir: str, annotations_file: str):
        """
        Initialize the data loader.
        
        Args:
            transcript_dir: Path to directory containing transcript CSV files
            annotations_file: Path to behavioral annotation codes CSV
        """
        self.transcript_dir = Path(transcript_dir)
        self.annotations_file = Path(annotations_file)
        
    def load_transcripts(self) -> pd.DataFrame:
        """
        Load all interview transcripts and combine with labels.
        
        Returns:
            DataFrame with columns: participant_id, question_id, question_text, 
            answer_text, combined_text, label
        """
        # Load annotation labels
        annotations = pd.read_csv(self.annotations_file)
        annotations.columns = ['participant_id', 'question_id', 'label']
        
        all_samples = []
        
        # Process each transcript file
        for transcript_file in sorted(self.transcript_dir.glob("P*_transcript.csv")):
            participant_id = transcript_file.stem.split('_')[0]
            
            # Load transcript
            df = pd.read_csv(transcript_file)
            df.columns = ['utterance_type', 'start_time', 'end_time', 'transcript']
            
            # Extract Q-A pairs
            questions = df[df['utterance_type'].str.startswith('Q', na=False)]
            answers = df[df['utterance_type'].str.startswith('A', na=False)]
            
            # Combine Q-A pairs
            for _, q_row in questions.iterrows():
                q_id = q_row['utterance_type']
                q_text = q_row['transcript']
                
                # Find corresponding answer
                a_id = q_id.replace('Q', 'A')
                a_row = answers[answers['utterance_type'] == a_id]
                
                if not a_row.empty:
                    a_text = a_row.iloc[0]['transcript']
                    
                    # Combine question and answer
                    combined_text = f"{q_text} {a_text}"
                    
                    all_samples.append({
                        'participant_id': participant_id,
                        'question_id': q_id,
                        'question_text': q_text,
                        'answer_text': a_text,
                        'combined_text': combined_text
                    })
        
        # Create DataFrame and merge with labels
        samples_df = pd.DataFrame(all_samples)
        samples_df = samples_df.merge(annotations, on=['participant_id', 'question_id'], how='inner')
        
        print(f"Loaded {len(samples_df)} samples from {len(samples_df['participant_id'].unique())} participants")
        print(f"Label distribution:\n{samples_df['label'].value_counts()}")
        
        return samples_df


class SyntacticFeatureExtractor:
    """
    Extracts syntactic features using count vectorization, TF-IDF, and POS tags.
    These features are interpretable and capture word usage patterns.
    """
    
    def __init__(self, max_features: int = 500):
        """
        Initialize syntactic feature extractors.
        
        Args:
            max_features: Maximum number of features for vectorizers
        """
        self.max_features = max_features
        self.count_vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        
    def extract_pos_features(self, text: str) -> Dict[str, float]:
        """
        Extract part-of-speech tag distribution features.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with POS tag proportions
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # Count POS tags
        pos_counts = {}
        for word, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        # Convert to proportions
        total = len(pos_tags) if pos_tags else 1
        pos_features = {
            'noun_ratio': (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                          pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total,
            'verb_ratio': (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                          pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) +
                          pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total,
            'adj_ratio': (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                         pos_counts.get('JJS', 0)) / total,
            'adv_ratio': (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                         pos_counts.get('RBS', 0)) / total,
            'pronoun_ratio': (pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)) / total,
        }
        
        return pos_features
    
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Fit vectorizers and extract all syntactic features.
        
        Args:
            texts: List of text documents
            
        Returns:
            Tuple of (count features, tfidf features, pos features DataFrame)
        """
        # Extract count and TF-IDF features
        count_features = self.count_vectorizer.fit_transform(texts).toarray()
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        
        # Extract POS features for each text
        pos_features_list = [self.extract_pos_features(text) for text in texts]
        pos_features_df = pd.DataFrame(pos_features_list)
        
        print(f"Extracted {count_features.shape[1]} count features")
        print(f"Extracted {tfidf_features.shape[1]} TF-IDF features")
        print(f"Extracted {pos_features_df.shape[1]} POS features")
        
        return count_features, tfidf_features, pos_features_df


class SemanticFeatureExtractor:
    """
    Extracts semantic features including sentiment scores and basic statistics.
    These features capture emotional tone and content characteristics.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment scores using VADER.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'sentiment_neg': scores['neg'],
            'sentiment_neu': scores['neu'],
            'sentiment_pos': scores['pos'],
            'sentiment_compound': scores['compound']
        }
    
    def extract_length_features(self, text: str) -> Dict[str, float]:
        """
        Extract length-based features (word count, sentence count, etc.).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with length features
        """
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def extract_all_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract all semantic features for a list of texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with all semantic features
        """
        all_features = []
        
        for text in texts:
            features = {}
            features.update(self.extract_sentiment_features(text))
            features.update(self.extract_length_features(text))
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        print(f"Extracted {features_df.shape[1]} semantic features")
        
        return features_df


class AdvancedFeatureExtractor:
    """
    Extracts advanced features using BERT embeddings.
    These capture contextual semantic meaning at a deeper level.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize BERT model and tokenizer.
        
        Args:
            model_name: Name of pretrained BERT model
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def extract_bert_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Extract BERT embeddings for a list of texts.
        Uses [CLS] token representation as sentence embedding.
        
        Args:
            texts: List of text documents
            max_length: Maximum sequence length for BERT
            
        Returns:
            Array of BERT embeddings (n_samples, 768)
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoded = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get BERT output
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        embeddings_array = np.array(embeddings)
        print(f"Extracted BERT embeddings: {embeddings_array.shape}")
        
        return embeddings_array


class FeaturePipeline:
    """
    Complete pipeline for extracting and combining all features.
    """
    
    def __init__(self, use_bert: bool = True):
        """
        Initialize all feature extractors.
        
        Args:
            use_bert: Whether to include BERT embeddings (slower but more powerful)
        """
        self.syntactic_extractor = SyntacticFeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor()
        self.use_bert = use_bert and BERT_AVAILABLE
        
        if use_bert and not BERT_AVAILABLE:
            print("Warning: BERT requested but PyTorch/Transformers not available. Skipping BERT features.")
        
        if self.use_bert:
            self.advanced_extractor = AdvancedFeatureExtractor()
    
    def extract_all_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract and combine all features.
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame with all extracted features
        """
        print("\n=== Extracting Syntactic Features ===")
        count_feat, tfidf_feat, pos_feat = self.syntactic_extractor.fit_transform(texts)
        
        print("\n=== Extracting Semantic Features ===")
        semantic_feat = self.semantic_extractor.extract_all_features(texts)
        
        # Combine basic features
        all_features = pd.concat([
            pd.DataFrame(tfidf_feat, columns=[f'tfidf_{i}' for i in range(tfidf_feat.shape[1])]),
            pos_feat,
            semantic_feat
        ], axis=1)
        
        # Add BERT embeddings if requested
        if self.use_bert:
            print("\n=== Extracting BERT Embeddings ===")
            bert_embeddings = self.advanced_extractor.extract_bert_embeddings(texts)
            bert_df = pd.DataFrame(
                bert_embeddings, 
                columns=[f'bert_{i}' for i in range(bert_embeddings.shape[1])]
            )
            all_features = pd.concat([all_features, bert_df], axis=1)
        
        print(f"\n=== Total Features Extracted: {all_features.shape[1]} ===")
        
        return all_features


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Define paths
    TRANSCRIPT_DIR = "Data/VetTrain_Transcripts/VetTrain_Transcripts"
    ANNOTATIONS_FILE = "Data/Behavioral Annotation Codes.csv"
    
    # Load data
    print("=" * 60)
    print("LOADING INTERVIEW DATA")
    print("=" * 60)
    loader = InterviewDataLoader(TRANSCRIPT_DIR, ANNOTATIONS_FILE)
    data = loader.load_transcripts()
    
    # Extract features
    print("\n" + "=" * 60)
    print("EXTRACTING FEATURES")
    print("=" * 60)
    
    # You can set use_bert=False for faster testing
    pipeline = FeaturePipeline(use_bert=False)  # Set to True for BERT features
    
    # Extract features from answer text only
    features = pipeline.extract_all_features(data['answer_text'].tolist())
    
    # Combine features with labels
    final_data = pd.concat([
        data[['participant_id', 'question_id', 'label']].reset_index(drop=True),
        features.reset_index(drop=True)
    ], axis=1)
    
    # Save to CSV
    output_file = "extracted_features.csv"
    final_data.to_csv(output_file, index=False)
    print(f"\n✓ Features saved to: {output_file}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(final_data)}")
    print(f"Total features: {features.shape[1]}")
    print(f"\nFeature types:")
    print(f"  - TF-IDF features: 500")
    print(f"  - POS features: 5")
    print(f"  - Sentiment features: 4")
    print(f"  - Length features: 5")
    if pipeline.use_bert:
        print(f"  - BERT embeddings: 768")
    
    print(f"\nLabel distribution:")
    print(final_data['label'].value_counts())
    
    print("\n✓ Feature extraction complete!")