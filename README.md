# 🎮 Movie Genre Prediction from Overview using NLP and ML

This project demonstrates how to build a machine learning model to predict the genre of a movie based on its overview using **Natural Language Processing (NLP)**.

---

## 📁 Step 1: Load the Dataset

```python
# Load genre data
genres_df = pd.read_csv('Movie Genre Dataset/movies_genres.csv')

# Load movie overview data
overview_df = pd.read_csv('Movie Genre Dataset/movies_overview.csv')
```

---

## 🗂️ Step 2: Map Genre IDs to Names

```python
# Create mapping from genre_id to genre_name
genre_mapping = dict(zip(genres_df['id'], genres_df['name']))

# Function to extract first genre from 'genre_ids' column
def map_genre(genre_id_str):
    try:
        genre_id = int(genre_id_str.split()[0])
    except Exception as e:
        return 'Unknown'
    return genre_mapping.get(genre_id, 'Unknown')

# Apply genre mapping to overview dataframe
overview_df['genre'] = overview_df['genre_ids'].apply(map_genre)
```

---

## 🪝 Step 3: Basic Cleaning

```python
# Drop rows where the overview is missing
overview_df = overview_df.dropna(subset=['overview'])
```

*Note: Advanced text cleaning (lowercasing, punctuation removal, stopwords, stemming/lemmatizing) can be added for further improvement.*

---

## 🔍 Step 4: Prepare Data for Training

```python
# Define features and target
X = overview_df['overview']
y = overview_df['genre']
```

Check genre distribution:

```python
print(y.value_counts())
```

---

## 🔀 Step 5: Split the Data

```python
from sklearn.model_selection import train_test_split

# Stratified split to preserve genre distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## 🤖 Step 6: Build and Train the Model

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)
```

---

## 📊 Step 7: Evaluate the Model

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions
y_pred = pipeline.predict(X_test)

# Accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Classification report
print(classification_report(y_test, y_pred))
```

---

## 📊 Step 8: Confusion Matrix

```python
# Get all known genres
all_genres = y.unique()

# Compute confusion matrix with all labels included
cm = confusion_matrix(y_test, y_pred, labels=all_genres)

# Plot using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_genres, yticklabels=all_genres)
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.title('Confusion Matrix of Genre Prediction')
plt.tight_layout()
plt.show()
```

---

## ✅ Optional Next Steps

- Improve text preprocessing (e.g., tokenization, lemmatization, stemming)
- Use advanced models like **Naive Bayes**, **SVM**, or **transformers (BERT)**
- Handle class imbalance using `class_weight='balanced'`
- Visualize most important TF-IDF features

---

## 🎓 Author
**Project by:** *[Your Name]*  
**Dataset:** Movie Genre Dataset  
**Tools Used:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK

---

