import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- Load Model ----
class OperatorClassifier(nn.Module):
    def __init__(self, input_dim):
        super(OperatorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 68)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- Extract Features ----
def extract_features(df, model, target_shape=(101, 50, 423)):
    df = df.astype(str)
    all_features = []

    for row in df.itertuples(index=False):
        row_features = []
        for cell in row:
            semantic_features = model.encode([cell], convert_to_tensor=False)[0]
            row_features.append(semantic_features)
        all_features.append(np.hstack(row_features))

    # Convert to NumPy and reshape correctly
    all_features = np.array(all_features, dtype=np.float32)
    target_rows, target_cols, feature_dim = target_shape
    feature_tensor = torch.tensor(all_features, dtype=torch.float32)

    needed_size = target_rows * target_cols * feature_dim
    feature_tensor = torch.flatten(feature_tensor)

    if feature_tensor.numel() < needed_size:
        padding_size = needed_size - feature_tensor.numel()
        padding = torch.zeros(padding_size, dtype=torch.float32)
        feature_tensor = torch.cat([feature_tensor, padding])
    else:
        feature_tensor = feature_tensor[:needed_size]  # Truncate if too large

    return feature_tensor.view(target_rows, target_cols, feature_dim)

# ---- Predict Operator ----
def predict_operator(test_file):
    print(f"🔍 Loading Test File: {test_file}")

    model = OperatorClassifier(input_dim=101*50*423)
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    df = pd.read_csv(test_file)
    df = df.sample(n=min(101, len(df)), replace=True)
    df = df.iloc[:, :50] if df.shape[1] > 50 else df

    features = extract_features(df, sentence_model, target_shape=(101, 50, 423))
    features_tensor = torch.tensor(features, dtype=torch.float32).view(1, -1)

    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        predicted_operator = "unstack" if probabilities[0] > probabilities[1] else "transpose"

    print(f"✅ Softmax Probabilities: {probabilities}")
    print(f"🔮 Predicted Operator: {predicted_operator}")

if __name__ == "__main__":
    predict_operator("testing_tables/high_blood_pressure.csv")  # Change filename


# import torch
# import pandas as pd
# import numpy as np
# from Train_model import TableCNN, apply_inverse_transformation

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load trained model
# model = TableCNN().to(DEVICE)
# model.load_state_dict(torch.load("trained_model.pth"))
# model.eval()

# def resize_dataframe(table):
#     """Ensures DataFrame is exactly (101,50) like training input."""
#     target_rows, target_cols = 101, 50
#     table = table[:target_rows, :target_cols]  # Trim if too large

#     # Fill missing rows
#     while table.shape[0] < target_rows:
#         table = np.vstack((table, table[:(target_rows - table.shape[0])]))

#     # Fill missing columns
#     while table.shape[1] < target_cols:
#         extra_cols = table[:, :target_cols - table.shape[1]]
#         table = np.hstack((table, extra_cols))

#     return table


# def test_table(file_path):
#     df = pd.read_csv(file_path)

#     # ✅ Resize table exactly like in training
#     resized_table = resize_dataframe(df.values)

#     # ✅ Convert entire table to strings (ensures dtype consistency)
#     resized_table = np.array(resized_table, dtype=str)

#     # ✅ Detect categorical columns
#     categorical_indices = []
#     for col in range(resized_table.shape[1]):
#         if not np.char.isnumeric(resized_table[:, col]).all():
#             categorical_indices.append(col)

#     print(f"✅ Categorical Columns Detected: {categorical_indices}")

#     # ✅ Encode categorical values safely
#     for col in categorical_indices:
#         unique_vals, encoded_vals = pd.factorize(resized_table[:, col])

#         # Ensure encoded_vals matches the original column shape
#         if len(encoded_vals) < resized_table.shape[0]:
#             extra_needed = resized_table.shape[0] - len(encoded_vals)
#             encoded_vals = np.pad(encoded_vals, (0, extra_needed), mode="constant", constant_values=-1)

#         resized_table[:, col] = encoded_vals  # Assign encoded values

#     print("✅ First 5 Rows After Encoding:")
#     print(resized_table[:5])  

#     # ✅ Convert to float32 after ensuring all values are numeric
#     resized_table_numeric = resized_table.astype(np.float32)

#     # ✅ Normalize data to range [0,1] to match training input
#     resized_table_numeric = (resized_table_numeric - np.min(resized_table_numeric)) / (np.max(resized_table_numeric) - np.min(resized_table_numeric) + 1e-8)

#     # Convert to PyTorch tensor
#     input_tensor = torch.tensor(resized_table_numeric, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     # ✅ Print input shape before passing to model
#     print(f"✅ Test Input Shape Before Model: {input_tensor.shape}")  

#     # Predict transformation
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_class = torch.argmax(output, dim=1).item()

#     transformation = "unstack" if predicted_class == 0 else "transpose"
#     print(f"🔮 Predicted Transformation: {transformation}")

# if __name__ == "__main__":
#     test_table("testing_tables/high_blood_pressure.csv")
