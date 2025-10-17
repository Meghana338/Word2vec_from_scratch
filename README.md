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
.  Navigate to the project directory:
    ```sh
    cd Word2Vec-from-Scratch(2)-GPU
    ```
3.  Launch Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
4.  Open `word2vec.ipynb` and follow one of the scenarios below.

---
### **Execution Scenarios**

This notebook supports different workflows depending on your goal.

#### Scenario A: Full Pipeline (Training from Scratch)
This is for running the entire process, from download to visualization. This is time-consuming.

* **Cells to Run:** Click **"Run All"** at the top of the notebook. The first cell will handle all installations and data downloads automatically.

#### Scenario B: Evaluation & Visualization Only (Using Pre-trained Vectors)
This is the recommended workflow for quickly exploring the project's results. This assumes the `word_vectors.npy` and `vocabulary.json` files are already created using the code.

1.  **Run Cell 1 ("Project Setup and Data Automation"):** This will install all required Python packages. You can ignore the data download messages if the files are already present.
2.  **SKIP** all the cells for data processing and model training (Cells under headings 1, 2, and 3).
3.  **Run all cells from the markdown header "4. Results and Analysis"** to the end of the notebook.
    * To run only the word analogy tests, execute the cells under section "4.2. Quantitative Analysis: Word Analogies".
    * To run only the visualizations, execute the cells under section "5. Visualizing the Embedding Space".

## The Project Journey & Key Decisions

This project involved several challenges that required iterative problem-solving.

1.  **Data Processing:** The initial plan to use the `wikiextractor` library failed due to Python version incompatibility. This was solved by building a more robust, self-contained parser using Python's built-in `xml` library, which also gracefully handled the discovery of a truncated (corrupted) XML file.

2.  **Tokenization:** The project initially used the `nltk` library, but a persistent and unresolvable `LookupError` in the execution environment forced a pivot. The solution was to **remove the NLTK dependency entirely** and replace its functions with a universal `re.findall()` approach, making the project more resilient.

3.  **GPU Acceleration:** Initial performance estimates on the CPU with NumPy predicted a training time of over a month. To make this feasible, I chose to accelerate the training with a GPU. I selected **CuPy** over PyTorch because it allowed me to keep the low-level, "from scratch" logic of the implementation while simply swapping the backend to the GPU.

4.  **Training Duration:** Even with a GPU, training on the full corpus was estimated to take many days. For this project, I made the practical decision to stop the training at **6.65 million pairs**. As the extensive evaluation results show, this was more than sufficient to produce a high-quality model that learned complex semantic relationships.
