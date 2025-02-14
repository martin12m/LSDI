# In this file you can extract the semantic features vector from the input file

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticFeatureExtractor:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f" Loading Semantic Model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

    def extract_semantic_features(self, cell):
        
        # Handle numeric values
        if isinstance(cell, (int, float)):
            return np.zeros(384)

        # Handle empty strings  
        cell = str(cell).strip() 

        if len(cell) == 0: 
            return np.zeros(384)

        embedding = self.model.encode([cell], convert_to_tensor=False)[0] 
        return embedding

    def extract_from_row(self, row):
        return np.array([self.extract_semantic_features(cell) for cell in row])


if __name__ == "__main__":
    semantic_extractor = SemanticFeatureExtractor()

    test_cases = [
        "Hello World",
        "This is a test sentence.",
        "123456",
        "email@domain.com",
        "www.google.com",
        "Hello123!",
        "",
        "AI and Machine Learning",
        "!!!",
        "Sentence embeddings are useful!"
    ]

    print("\n Testing Semantic Feature Extraction:\n")
    for test in test_cases:
        print(f"Input: {test}")
        semantic_features = semantic_extractor.extract_semantic_features(test)
        print(f"First 10 Semantic Features: {semantic_features[:10]}\n")  