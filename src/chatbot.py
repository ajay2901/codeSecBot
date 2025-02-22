import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import TextPreprocessor
from .logger import Logger
import csv
import os
from datetime import datetime

class ChatBot:
    def __init__(self, data_path, log_path="logs/unknown_questions.csv"):
        self.df = pd.read_csv(data_path)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer()
        self.logger = Logger()  # Initialize logger
        self._prepare_data()
    
    def _setup_logging(self):
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Create log file with header if new
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "original_query", "processed_query", "similarity_score"])
    

    def log_unknown_question(self, query, processed_query, score):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, query, ' '.join(processed_query), score])

        
    def _prepare_data(self):
        # Preprocess all questions
        self.df['Processed'] = self.df['Question'].apply(
            lambda x: ' '.join(self.preprocessor.preprocess(x))
        )
        
        # Train TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Processed'])
        
    def get_response(self, query, threshold=0.5):
        # Preprocess user input
        processed_query = ' '.join(self.preprocessor.preprocess(query))
        
        # Vectorize query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        best_match_idx = similarities.argmax()
        max_score = similarities[0, best_match_idx]
        
        if max_score > threshold:
            response =  self.df.iloc[best_match_idx]['Answer']
        else:
            response =  "I'm not sure how to help with that. Can you rephrase?"
            self.logger.log_unknown(query, processed_query, max_score)

        
        # Log ALL interactions
        self.logger.log_interaction(
            query, 
            processed_query, 
            response, 
            max_score
        )

        return response