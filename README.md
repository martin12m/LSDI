# **AutoTables: Synthesizing Multi-step Transformations to Relationalize Tables**

## **University**

**Technische Universität Berlin (TU Berlin)**  
**Winter Semester 2024/2025**

## **Module**

**Large Scale Data Integration Project**

## **Project Title**

**AutoTables: Synthesizing Multi-Step Transformations to Relationalize Tables**

---

## **Project Description**

### **Overview**

AutoTables is an AI-driven system that **automatically transforms non-relational tables into relational tables** without requiring human-labeled examples. It synthesizes **multi-step transformations**, enabling seamless querying and integration into structured database environments.

The system leverages **deep learning techniques, convolutional neural networks (CNNs), and self-supervised training** to predict and apply table transformations. It processes raw tabular data, applies transformations, and ranks the best results for relationalization.

### **Key Features**

- **Automatic Table Transformation:** Detects and applies the most relevant transformations (e.g., **unstack, transpose**).
- **Deep Learning-based Feature Extraction:** Learns meaningful **syntactic & semantic** patterns from tables.
- **Self-Supervised Learning:** Generates its own training dataset via **inverse transformations** (e.g., stack ↔ unstack).
- **CNN-inspired Architecture:** Uses convolutional layers to analyze **structural patterns** in tabular data.

---

## **System Architecture**

The AutoTables model consists of four main layers:

### 1️⃣ **Embedding Layer**

- Extracts **syntactic features** (e.g., character count, digit ratio, punctuation).
- Extracts **semantic features** using **Sentence-BERT**.
- Produces a **423-dimensional feature vector per cell**.

### 2️⃣ **Dimension Reduction Layer**

- Uses **two 1×1 convolutions** to shrink feature vectors **(423 → 64 → 32 per cell)**.
- Reduces computational complexity for further processing.

### 3️⃣ **Feature Extraction Layer**

- **Column-wise and Row-wise Feature Extraction:**
  - **Column:** Uses **(1×1, 1×2) filters** to detect column dependencies.
  - **Row:** Uses **(1×1, 2×1) filters** to capture row relationships.
- **Global & Local Feature Pooling:**
  - Uses **Average Pooling** to summarize column-wise and row-wise information.
  - Produces **structural embeddings** to guide relational transformations.

### 4️⃣ **Output Layer**

- Fully connected layers classify the **best transformation** (e.g., Unstack, Transpose).
- Generates **probability scores** for ranking transformations.
- Uses **Softmax** to normalize predictions.

---

## **Folder Structure**

```
📦 AutoTables
 ┣ 📂 datasets
 ┃ ┣ 📂 testing_data                 # Relational input tables (CSV files)
 ┃ ┣ 📂 transformed_data             # Non-relational tables (after applying transformations)
 ┃ ┣ 📂 resized_data                 # Resized tables (after fixing dimensions)
 ┃ ┗ 📜 sample.csv                   # Example dataset
 ┣ 📂 model_architecture
 ┃ ┣ 📜 embedding_layer.py            # Extracts syntactic & semantic features
 ┃ ┣ 📜 dimension_reduction.py        # Applies 1×1 convolutions to reduce dimensions
 ┃ ┣ 📜 feature_extraction.py         # Extracts table structure using CNN
 ┃ ┣ 📜 output_layer.py               # Predicts best transformation
 ┃ ┣ 📜 semantic_features.py          # Debug the featuer vector values by appyling SemanticFeatures
 ┃ ┣ 📜 synthetic_features.py          # Debug the featuer vector values by appyling SyntheticFeatures
 ┃ ┗ 📜 utils.py                      # Helper functions for preprocessing
 ┣ 📜 Train_the_Model.py               # script to train AutoTables
 ┣ 📜 test_the_Model.py               # script to test AutoTables
 ┣ 📜 requirements.txt                 # Python dependencies
 ┣ 📜 README.md                        # Project documentation
 ┗ 📜 LICENSE                          # License file
```

---

## **Installation & Setup**

### **1️⃣ Prerequisites**

Ensure you have **Python 3.8+** and the following dependencies installed:

```bash
pip install -r requirements.txt
```

### **2️⃣ Dataset Preparation**

- Place **relational input tables** in **datasets/testing_data/**.
- The model will automatically generate transformed tables in **datasets/transformed_data/**.
- The resized tables will be stored in **datasets/resized_data/**.

### **3️⃣ Running the Model**

To **train** the model:

```bash
python Train_the_Model.py
```

To **test** on a new dataset:

```bash
python Test_the_Model.py
```

---

## **Evaluation & Results**

### **Key Metrics**

✔ **Hit@1**: 57% of transformations are correctly predicted on the first try.  
✔ **Hit@3**: 75% accuracy when considering the top-3 predictions.

---

## **Acknowledgments**

This project is inspired by the research paper:  
📄 **"Auto-Tables: Synthesizing Multi-Step Transformations to Relationalize Tables"**  
Authors: **Peng Li, Yeye He, Cong Yan, Yue Wang, Surajit Chaudhuri**  
Published in **PVLDB 2023**

---

## **Contributors**

| Name               |
| ------------------ |
| **Ankit Mavani**   |
| **Victor Aguiar**  |
| **Martin Manolov** |

---
