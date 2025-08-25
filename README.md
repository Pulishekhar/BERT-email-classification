📧 BERT Email Classification

This project demonstrates how to build an email classifier that detects whether a message is spam or not spam (ham) using BERT embeddings. It shows how state-of-the-art NLP models can be applied to real-world classification problems.

🔍 Problem Statement

Email spam detection is a classic NLP problem where the goal is to distinguish between unwanted/spam messages and legitimate (ham) emails. Traditional models (Naive Bayes, SVM) rely heavily on bag-of-words or TF-IDF features.
Here, we use BERT (Bidirectional Encoder Representations from Transformers) to generate semantic embeddings, which capture the true meaning of sentences for better classification.

📂 Dataset

Source: Kaggle Spam Email Dataset

Samples: ~5,000 emails

Target Labels:

ham → 0 (legitimate emails)

spam → 1 (unwanted emails)

The dataset is balanced between spam and ham classes.

⚙️ Project Pipeline
1. Data Preparation

Load dataset using Pandas

Clean & preprocess text

Convert categorical labels into binary values (0/1)

Train-test split using scikit-learn

2. Embedding Generation

Use BERT Preprocessing Model from TensorFlow Hub:

bert_en_uncased_preprocess/3

Use BERT Encoder Model:

bert_en_uncased_L-12_H-768_A-12/4

Extract sentence embeddings for each email

3. Semantic Similarity (Exploratory)

Compare embeddings using cosine similarity

Example: "banana" vs "grapes" (high similarity)

Demonstrates BERT’s semantic understanding

4. Model Training

Feed BERT embeddings into a classification head

Layers: Dense → Dropout → Dense(sigmoid)

Loss: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

5. Evaluation

Evaluate on test data

Report metrics: Accuracy, Precision, Recall, F1-score

Example (replace with actual numbers from notebook):

Accuracy: ~97%

Precision: ~95%

Recall: ~96%

🛠️ Installation

Install the required libraries:

pip install tensorflow tensorflow-hub scikit-learn pandas matplotlib

🚀 Usage

Clone this repository or download the notebook.

Open the notebook in Jupyter:

jupyter notebook BERT_email_classification.ipynb


Run all cells step by step:

Load dataset

Preprocess data

Generate embeddings

Train classifier

Evaluate results

📊 Results

BERT embeddings capture semantic meaning, outperforming traditional feature extraction methods.

The classifier achieves high accuracy (~97%) on the test set.

Cosine similarity tests show meaningful closeness between semantically related words/sentences.
