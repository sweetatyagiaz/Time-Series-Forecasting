# Naive Bayes

## Overview

Naive Bayes is a family of supervised machine learning algorithms based on Bayes' Theorem and the assumption that features are conditionally independent given the target class.

Despite its simplicity, Naive Bayes performs remarkably well in many real-world classification tasks, particularly in text classification, spam filtering, sentiment analysis, document categorization, and recommendation systems.

Key characteristics:

- Fast training and inference
- Works well with high-dimensional data
- Effective with small training datasets
- Scalable to large datasets
- Strong baseline classification algorithm

---

# Bayes Theorem

Naive Bayes is based on Bayes' Theorem:

\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]

For classification:

\[
P(y|X) = \frac{P(X|y)P(y)}{P(X)}
\]

Where:

- \(P(y|X)\) = Posterior Probability
- \(P(X|y)\) = Likelihood
- \(P(y)\) = Prior Probability
- \(P(X)\) = Evidence

The model predicts the class with the highest posterior probability:

\[
\hat{y} = \arg\max_y P(y)\prod_{i=1}^{n}P(x_i|y)
\]

The "naive" assumption is that all features are independent given the class label.

---

# Implemented Models

## Gaussian Naive Bayes (GaussianNB)

Assumes features follow a Gaussian (Normal) distribution.

### Best For

- Continuous numerical data
- Sensor measurements
- Medical datasets
- Financial indicators

### Example Features

```text
Age
Salary
Temperature
Stock Returns
```

### Advantages

- Fast
- Handles continuous features directly
- Works well with small datasets

### Limitations

- Assumes normal distribution
- Sensitive to non-Gaussian data

---

## Multinomial Naive Bayes (MultinomialNB)

Designed for discrete count-based features.

Commonly used in Natural Language Processing.

### Best For

- Text Classification
- Spam Detection
- Sentiment Analysis
- News Categorization

### Example Features

```text
Word Counts
Document Frequencies
TF-IDF Features
```

### Advantages

- Excellent for text analytics
- Efficient on sparse matrices
- Highly scalable

### Limitations

- Assumes count data
- Less suitable for continuous variables

---

## Bernoulli Naive Bayes (BernoulliNB)

Designed for binary-valued features.

Each feature represents presence or absence.

### Best For

- Binary Classification
- Keyword Detection
- Short Text Analysis

### Example Features

```text
Contains Keyword = Yes/No
Clicked Ad = Yes/No
Purchased Product = Yes/No
```

### Advantages

- Simple implementation
- Effective for binary features

### Limitations

- Ignores frequency information
- Not suitable for count-based data

---

## Complement Naive Bayes (ComplementNB)

Designed to improve classification performance on imbalanced datasets.

Uses statistics from the complement of each class.

### Best For

- Imbalanced Text Datasets
- Document Classification
- Large Sparse Feature Sets

### Advantages

- Better handling of class imbalance
- Improved performance on skewed datasets

### Limitations

- Primarily useful for text classification

---

## Categorical Naive Bayes (CategoricalNB)

Designed for categorical features.

Each feature follows a categorical distribution.

### Best For

- Survey Data
- Customer Segmentation
- Demographic Analysis

### Example Features

```text
Gender
Education Level
Occupation
Region
```

### Advantages

- Naturally handles categorical variables
- Simple and interpretable

### Limitations

- Requires categorical encoding

---

# Model Selection Guide

| Data Type | Recommended Model |
|------------|------------------|
| Continuous Numerical Data | GaussianNB |
| Word Counts | MultinomialNB |
| Binary Features | BernoulliNB |
| Imbalanced Text Data | ComplementNB |
| Categorical Features | CategoricalNB |

---

# Applications

## Natural Language Processing

- Spam Detection
- Sentiment Analysis
- Intent Classification
- Document Classification
- Topic Categorization

## Finance

- Credit Risk Assessment
- Fraud Detection
- Customer Segmentation

## Healthcare

- Disease Prediction
- Clinical Diagnosis Support

## Cybersecurity

- Malware Detection
- Intrusion Detection
- Threat Classification

## Marketing

- Customer Classification
- Lead Scoring
- Campaign Response Prediction

---

# Advantages

- Fast Training
- Fast Prediction
- Works Well with High-Dimensional Data
- Requires Less Training Data
- Easy to Interpret
- Highly Scalable
- Excellent Baseline Classifier

---

# Limitations

- Strong Independence Assumption
- Cannot Model Feature Interactions
- Probability Estimates May Be Poorly Calibrated
- Performance May Degrade on Complex Nonlinear Problems

---

# Computational Complexity

| Operation | Complexity |
|------------|------------|
| Training | O(n × d) |
| Prediction | O(k × d) |
| Memory Usage | O(k × d) |

Where:

- n = Number of Samples
- d = Number of Features
- k = Number of Classes

---

# Scikit-Learn Example

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")
```

---

# Hyperparameters

## GaussianNB

| Parameter | Description |
|------------|-------------|
| var_smoothing | Portion of largest variance added for stability |

## MultinomialNB

| Parameter | Description |
|------------|-------------|
| alpha | Additive smoothing parameter |
| fit_prior | Learn class prior probabilities |

## BernoulliNB

| Parameter | Description |
|------------|-------------|
| alpha | Laplace smoothing |
| binarize | Threshold for binarization |

---

# Evaluation Metrics

Common evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss
- Confusion Matrix

---

# Best Practices

- Normalize or scale data when appropriate
- Remove highly correlated features if possible
- Use Laplace smoothing to avoid zero probabilities
- Evaluate class imbalance before selecting a variant
- Compare against tree-based and ensemble methods

---

# When to Use Naive Bayes

Use Naive Bayes when:

✅ Fast training is required

✅ Dataset is high-dimensional

✅ Text classification is involved

✅ Limited training data is available

✅ A strong baseline model is needed

Avoid Naive Bayes when:

❌ Features are highly dependent

❌ Complex nonlinear relationships dominate

❌ High-quality probability calibration is required

---

# References

- Scikit-Learn Documentation
- Pattern Recognition and Machine Learning — Christopher Bishop
- Machine Learning: A Probabilistic Perspective — Kevin Murphy
- Introduction to Information Retrieval — Manning, Raghavan, Schütze