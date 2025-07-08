# Naive Bayes Classifier for Sentiment Analysis (IMDB Dataset)

This project implements a Naive Bayes classifier for sentiment analysis on movie reviews from the IMDb dataset. The classifier is built from scratch to demonstrate fundamental concepts of natural language processing (NLP), feature selection, and probabilistic modeling.

---

## Contents

- Vocabulary creation using Information Gain  
- Text-to-feature vector conversion (bag-of-words)  
- Custom Naive Bayes classifier implementation  
- Comparison with Scikit-learn `BernoulliNB`  
- Model evaluation with precision, recall, F1, and accuracy metrics  

---

## Code Structure

### `WordStat`
Tracks word statistics: counts in positive/negative documents.

### `VocabularyBuilder`
Implements vocabulary creation:
- Filters words (removes most frequent and least frequent)
- Calculates Information Gain (IG) for feature selection
- Generates final vocabulary with `m` features

### `FeatureVector`
Converts documents to feature vectors:
- Binary features (1 = word present, 0 = absent)
- Appends sentiment label (1 = positive, 0 = negative)

### `NaiveBayes`
Custom Naive Bayes implementation:
- Uses log probabilities with Laplace smoothing
- Calculates log-likelihood for each feature
- Predicts most probable class for new documents

---

## Execution Instructions

1. Download the IMDB dataset  
2. Unzip the dataset to your project directory  
3. Run the Jupyter notebook:

```bash
jupyter notebook NaiveBayesClassifier.ipynb
```
## Execution Pipeline

### Vocabulary Creation
 Parameters: `n=50`, `k=80`, `m=500`
- Removes top-50 most frequent words and 80 least frequent words
- Selects top-500 words based on Information Gain

### Feature Vector Creation
- Converts all reviews into binary feature vectors

### Training & Evaluation
- Trains custom Naive Bayes classifier
- Trains Scikit-learn BernoulliNB for comparison
- Prints precision, recall, F1-score, and accuracy metrics

## Model Performance Comparison
 |Model                 | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| Custom NaiveBayes          | 0.85     | 0.84      | 0.85   | 0.84     |
| Scikit-Learn NaiveBayes    | 0.86     | 0.85      | 0.86   | 0.85     |

## Requirements
- Python 3.7+
- Libraries:
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `sklearn`
 
Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn jupyter
```
--- 
## Key Features
### Information Gain Calculation
Statistical method for selecting the most discriminative features

### Laplace Smoothing
Handles unseen words during classification

### Efficient Vectorization
Binary features optimized for memory and performance

### Visual Analytics
Learning curves, confusion matrices, and feature importance plots

---

## Sample Output

```text
Vocabulary creation completed (500 words)  
First 10 vocabulary words: ['excellent', 'wonderful', 'best', 'perfect', 'great', ...]

Training set size: 25,000 vectors  
Test set size: 25,000 vectors

Custom NaiveBayes Results:
              precision    recall  f1-score   support

    Negative       0.84      0.85      0.84     12500
    Positive       0.85      0.84      0.84     12500

    accuracy                           0.85     25000

Scikit-Learn Results:
              precision    recall  f1-score   support

    Negative       0.85      0.86      0.85     12500
    Positive       0.86      0.85      0.85     12500

    accuracy                           0.86     25000

```
---

## Customization

Modify these parameters in the notebook:

```python
# Vocabulary parameters
n = 50    # Ignore top-n frequent words
k = 80    # Ignore bottom-k rare words
m = 500   # Select top-m features

# Training parameters
train_size = 25000  # Training samples per class
test_size = 25000   # Test samples per class
```
