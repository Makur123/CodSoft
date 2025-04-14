# 🎬 Movie Genre Prediction from Overview Text

This project demonstrates how to build a machine learning model that predicts the genre of a movie based on its overview (plot summary). The dataset includes movie overviews and genre IDs, which are mapped to human-readable genre names. The process includes data loading, cleaning, feature extraction, training, and evaluation.

---

## 1. Project Setup

### 1.1. Dependencies
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### 1.2. NLTK Resource Setup
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.data.path.append(r'C:\Users\dhuor\nltk_data')  # Adjust path as needed
```

---

## 2. Load and Prepare Dataset

### 2.1. Load Data
```python
genres_df = pd.read_csv('Movie Genre Dataset/movies_genres.csv')
overview_df = pd.read_csv('Movie Genre Dataset/movies_overview.csv')
```

### 2.2. Genre Mapping
```python
genre_mapping = dict(zip(genres_df['id'], genres_df['name']))

def map_genre(genre_id_str):
    try:
        genre_id = int(genre_id_str.split()[0])
    except:
        return 'Unknown'
    return genre_mapping.get(genre_id, 'Unknown')

overview_df['genre'] = overview_df['genre_ids'].apply(map_genre)
overview_df.dropna(subset=['overview'], inplace=True)
```

---

## 3. Feature Extraction and Model Training

### 3.1. Split Dataset
```python
X = overview_df['overview']
y = overview_df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### 3.2. Pipeline and Training
```python
if y_train.nunique() < 2:
    print(f"Insufficient classes for training. Found classes: {y_train.unique()}.")
else:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
```

---

## 4. Evaluation

### 4.1. Accuracy & Report
```python
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.2f}')
    print(classification_report(y_test, y_pred))
```

### 4.2. Confusion Matrix
```python
    all_labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
```

---

## 5. Notes and Future Work

- The genre mapping step can result in many entries labeled as 'Unknown'. Consider enhancing the genre parsing logic if multiple genre IDs are present.
- Optional preprocessing like text cleaning (stopword removal, lemmatization) can be added for more refined models.
- Try advanced NLP models like BERT or other transformer-based architectures for improved performance.

---

## 📁 Folder Structure
```
Movie Genre Dataset/
├── movies_genres.csv
├── movies_overview.csv
├── confusion_matrix.png
```

---

## ✅ Conclusion
This pipeline successfully demonstrates how to classify movie genres based on text summaries using classical NLP and machine learning techniques in Python with Scikit-learn.

