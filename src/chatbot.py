import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import TextPreprocessor

class ChatBot:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer()
        self._prepare_data()
        
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
            return self.df.iloc[best_match_idx]['Answer']
        return "I'm not sure how to help with that. Can you rephrase?"