# src/logger.py
import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self._setup_logs()
        
    def _setup_logs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize all_interactions.csv
        if not os.path.exists(f"{self.log_dir}/all_interactions.csv"):
            with open(f"{self.log_dir}/all_interactions.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "original_query", 
                    "processed_query", 
                    "response", 
                    "similarity_score"
                ])
                
        # Initialize unknown_questions.csv
        if not os.path.exists(f"{self.log_dir}/unknown_questions.csv"):
            with open(f"{self.log_dir}/unknown_questions.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "original_query", 
                    "processed_query", 
                    "similarity_score"
                ])

    def log_interaction(self, query, processed_query, response, score):
        self._write_log(
            "all_interactions.csv",
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                query,
                ' '.join(processed_query),
                response,
                f"{score:.4f}"
            ]
        )

    def log_unknown(self, query, processed_query, score):
        self._write_log(
            "unknown_questions.csv",
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                query,
                ' '.join(processed_query),
                f"{score:.4f}"
            ]
        )

    def _write_log(self, filename, row):
        with open(f"{self.log_dir}/{filename}", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)