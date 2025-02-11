import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticFeatureExtractor:
    """
    Extracts semantic features using Sentence-BERT (384-dimensional embeddings).
    - Handles missing values, numeric inputs, and special cases.
    - Converts all input to strings before embedding.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"✅ Loading Semantic Model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

    def extract_semantic_features(self, cell):
        """
        Extracts a 384-dimensional semantic embedding from a cell.
        - If the cell is numeric, it returns a zero vector (no embedding needed).
        - If the cell is an empty string, it also returns a zero vector.
        """
        if isinstance(cell, (int, float)):  # If numeric, return zero embedding
            return np.zeros(384)

        cell = str(cell).strip()  # Convert to string and remove whitespace

        if len(cell) == 0:  # Handle empty values
            return np.zeros(384)

        embedding = self.model.encode([cell], convert_to_tensor=False)[0]  # Extract BERT embedding
        return embedding

    def extract_from_row(self, row):
        """
        Extracts semantic features for each cell in a row.
        """
        return np.array([self.extract_semantic_features(cell) for cell in row])

# -------------------
# Example Usage:
# -------------------
if __name__ == "__main__":
    # Initialize semantic model
    semantic_extractor = SemanticFeatureExtractor()

    # Test cases
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

    print("\n🔍 Testing Semantic Feature Extraction:\n")
    for test in test_cases:
        print(f"📝 Input: {test}")
        semantic_features = semantic_extractor.extract_semantic_features(test)
        print(f"🔢 First 10 Semantic Features: {semantic_features[:10]}\n")  # Print first 10 values