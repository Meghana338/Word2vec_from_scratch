# 🧠 Word2Vec from Scratch with GPU Acceleration

This repository presents a **complete from-scratch implementation** of the **Word2Vec (Skip-Gram)** model in Python.  
The goal is to gain a deep, conceptual understanding of how word embeddings are learned — by manually building each step of the pipeline, from **data ingestion** to **training** and **evaluation**.

The implementation leverages **CuPy** for **GPU acceleration**, allowing efficient large-scale training on datasets such as **`enwik9`**.

---

## 🚀 Project Overview

This project implements the **Word2Vec Skip-Gram** model entirely from first principles — no high-level machine learning or NLP libraries are used.  
Every stage, including **preprocessing**, **vocabulary creation**, **skip-gram pair generation**, **model optimization**, and **embedding visualization**, is developed manually.

### **Model Details**
- **Architecture:** Skip-Gram  
- **Objective:** Predict context words from a target word  
- **Optimization:** Negative Sampling (approximated via full softmax)  
- **Corpus:** `enwik9` (English Wikipedia extract)

### **Technology Stack**
- **Language:** Python 3  
- **Core Library:** [CuPy](https://cupy.dev/) — GPU-accelerated numerical computing (NumPy-compatible)  
- **Environment:** Jupyter Notebook  

---

## 🧩 1. Data Ingestion and Preprocessing

### **1.1 Corpus Selection**
The **enwik9** dataset — a billion-character English Wikipedia dump — was selected for its linguistic diversity and contextual richness, ideal for learning meaningful word relationships.

### **1.2 XML Parsing and Cleaning**
Since the corpus is provided in XML with Wikitext markup, a custom preprocessing pipeline was developed:

- **Stream-Based Parsing:**  
  Used `xml.etree.ElementTree.iterparse()` for incremental XML parsing, enabling memory-efficient processing of large files.

- **Markup Removal:**  
  Regular expressions were applied to clean Wikipedia-specific syntax (`[[links]]`, `{{templates}}`, HTML tags), producing clean, plain text ready for tokenization and vocabulary building.

---

## ⚙️ 2. Model Training

The Skip-Gram model was implemented manually with:
- **Forward and backward propagation** using CuPy arrays  
- **Softmax / Negative Sampling** for efficient probability estimation  
- **Custom gradient updates** for embedding matrices  

Training was performed over the cleaned corpus, gradually refining embeddings through stochastic updates.

---

## 📈 3. Evaluation: Relational Semantics

To assess semantic understanding, the model was tested on classic **word analogy tasks** such as _“king – man + woman = queen”_.  
These tasks evaluate how well vector arithmetic captures relational meaning.

### ✅ **Successes**
The model performed strongly on structured relationships, especially gender and geography:

- **king – man + woman → queen**  
- **he – man + woman → she**  
- **japan – tokyo + beijing → china**

These results show that the embedding space captures **consistent semantic structures**, modeling gender and country–capital analogies effectively.

### ⚠️ **Partial Successes and Failures**

Some analogies exposed the model’s limitations and corpus-driven biases:

- **france – paris + london → spain**  
  Expected: *england* or *britain*. Reflects historical ties between France and Spain present in Wikipedia data.  

- **walking – walk + swim → 1692**  
  Fails to model **verb tense** patterns — finer grammatical nuances often require syntax-aware models.  

- **japan – yen + dollar → china**  
  Captures a general *country–economy* relationship but misses the specific currency mapping.

---

## 🎨 4. Visualization and Output

The learned embeddings are visualized using **t-SNE** and **PCA**, revealing clusters of semantically related words.  
Embeddings are exported in both `.npy` and `.txt` formats for downstream NLP tasks.

```bash
embeddings.npy     # NumPy binary file  
embeddings.txt     # Human-readable text file
```

---

## 📂 Repository Structure

```
Word2vec_from_scratch/
│
├── data/                 # Corpus and preprocessing scripts
├── src/
│   ├── preprocess.py      # XML parsing and cleaning
│   ├── train.py           # Skip-Gram training loop
│   ├── model.py           # Word2Vec architecture and updates
│   ├── evaluate.py        # Analogy and similarity testing
│   └── visualize.py       # Embedding visualization (t-SNE/PCA)
│
├── embeddings/
│   ├── embeddings.npy
│   └── embeddings.txt
│
├── README.md
└── requirements.txt
```

---

## 🔗 Commit History
For detailed progress and code evolution, visit the commit history:

👉 [View Commits by Meghana338](https://github.com/Meghana338/Word2vec_from_scratch/commits?author=Meghana338)

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

### 💡 Author
**Meghana338**  
GPU-Accelerated Word Embedding Researcher & Developer  
[GitHub Repository](https://github.com/Meghana338/Word2vec_from_scratch)
