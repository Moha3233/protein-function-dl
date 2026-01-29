# 🧬 Protein Function Classification Using Deep Learning (PyTorch)

## 📌 Overview
Accurate protein function annotation remains a fundamental challenge in computational biology. This project presents a **deep learning–based framework** for **protein function classification** using **sequence-derived features**, implemented entirely in **PyTorch**.

The study focuses on:
- **Binary classification**: Enzyme vs Non-enzyme  
- (Optional) **Multi-class classification**: Protein functional families  

The workflow simulates a **real-world computational biology research pipeline**, including data preprocessing, feature engineering, model training, evaluation, and biological interpretation.

---

## 🎯 Objectives
- Build an end-to-end **protein function classification system**
- Apply **deep learning (MLP)** using PyTorch
- Compare performance with **classical ML models**
- Provide **biological interpretability**
- Ensure **reproducibility and clean code organization**

---

## 🧪 Dataset
**Source options**
- UniProt (Swiss-Prot reviewed protein sequences)
- Kaggle protein sequence datasets with functional labels

**Data fields**
- Amino acid sequence
- Functional label  
  - Binary: Enzyme (EC number present) / Non-enzyme  
  - Multi-class: Functional family

**Preprocessing**
- Remove sequences:
  - Length < 50 amino acids
  - Ambiguous residues (X, B, Z, J, U, O)
- Handle class imbalance
- Data split:
  - 70% Train / 15% Validation / 15% Test

---

## 🧠 Feature Engineering

### Amino Acid Composition (AAC)
- Normalized frequency of 20 amino acids
- Captures global biochemical properties  
- **20 features**

### Dipeptide Composition (DPC)
- Frequency of all possible amino acid pairs (20 × 20)
- Captures local sequence-order information  
- **400 features**

### Final Feature Vector
- AAC (20) + DPC (400) = **420 features**
- Standardized using `StandardScaler`

---

## 🏗️ Model Architecture

### Deep Learning Model (Primary)
**Multi-Layer Perceptron (MLP)**

**Architecture**
- Input: 420 neurons
- Hidden layers:
  - Dense (256) → ReLU → Dropout (0.3)
  - Dense (128) → ReLU → Dropout (0.3)
- Output:
  - Binary: Sigmoid
  - Multi-class: Softmax

**Training**
- Loss:
  - Binary Cross-Entropy
  - Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 30–50
- Early stopping on validation loss

---

## 🔁 Custom Training Loop
- Manual forward pass
- Loss computation
- Backpropagation
- Weight updates
- Validation without gradients
- Early stopping

✔️ No high-level training wrappers used

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (binary)

### Expected Performance
| Model | Accuracy |
|------|---------|
| Random Forest | ~65–68% |
| Deep Learning (MLP) | **75–80%** |

---

## 🧬 Biological Interpretation
- Enzymes enriched in catalytic residues (H, D, E, C)
- Distinct dipeptide patterns reflect active-site geometry
- Non-enzymes show structural/binding-related composition

**Applications**
- Functional annotation
- Enzyme discovery
- Protein engineering

---

## 🔬 Comparative Study
Classical ML models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

**Comparison**
- Performance
- Training time
- Interpretability

Deep learning captures **non-linear feature interactions** missed by classical models.

---

## 📁 Project Structure
protein-function-dl/
│
├── data_preprocessing.py
├── feature_engineering.py
├── model.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── results/

yaml
Copy code

---

## 🔁 Reproducibility
- Fixed random seeds
- Modular codebase
- Deterministic preprocessing

---

## 📦 Requirements
torch
numpy
pandas
scikit-learn

yaml
Copy code

---

## ⚠️ Limitations
- Handcrafted features miss long-range dependencies
- No structural or evolutionary information

---

## 🚀 Future Scope
- Transformer embeddings (ProtBERT, ESM)
- Attention-based interpretability
- Multi-task learning (EC number + family)
- Structural feature integration

---

## 🏁 Conclusion
This project demonstrates a **reproducible, interpretable deep learning framework** for protein function classification using only primary sequence information, suitable for **PhD-level computational biology research**.
✅ You’re doing this exactly right
