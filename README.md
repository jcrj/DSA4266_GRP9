# DSA4266_GRP9

# Cyberbullying Detection Using Machine Learning and Deep Learning Techniques

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
  - [Machine Learning Models](#machine-learning-models)
  - [Deep Learning Models](#deep-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)
- [How to Run the Code](#how-to-run-the-code)
- [References](#references)

## Introduction

Cyberbullying has become a pervasive issue with the rise of social media platforms. Detecting cyberbullying content automatically can help mitigate its harmful effects. This project aims to develop and compare various machine learning and deep learning models to classify tweets as cyberbullying or not.

## Dataset

The dataset used in this project is the **Cyberbullying Tweets Dataset**, which contains tweets labeled with different types of cyberbullying or as not cyberbullying. The dataset includes the following columns:

- `tweet_text`: The text content of the tweet.
- `cyberbullying_type`: The category of cyberbullying or `not_cyberbullying`.

**Classes in the Dataset:**

- `religion`
- `age`
- `ethnicity`
- `gender`
- `not_cyberbullying`
- `other_cyberbullying`

## Exploratory Data Analysis (EDA)

- **Label Distribution:** Analyzed the spread of different labels to understand class imbalance.
- **Word Cloud:** Generated a word cloud to visualize the most frequent words in the tweets.
- **Top Words per Label:** Identified the most common words associated with each label.

## Data Preprocessing

Text data from social media often contains noise such as URLs, emojis, HTML tags, and colloquial terms. The preprocessing steps included:

1. **Removing Duplicates:** Eliminated duplicate tweets to reduce redundancy.
2. **Text Cleaning:**
   - Removed URLs, HTML tags, and emojis.
   - Converted text to lowercase.
   - Removed punctuation and non-ASCII characters.
   - Tokenized text and removed stopwords.
   - Optionally performed lemmatization to reduce words to their base form.

## Feature Engineering

Several feature extraction methods were employed to convert text into numerical representations:

1. **Word Embeddings:**
   - **Word2Vec:** Trained on the dataset to capture semantic relationships.
   - **GloVe:** Pre-trained embeddings loaded for comparison.

2. **Vectorizers:**
   - **TF-IDF Vectorizer:** Captures the importance of words in the document relative to the corpus.
   - **Count Vectorizer:** Counts the frequency of words in the text.

## Modeling

### Machine Learning Models

1. **Logistic Regression:** Used as a baseline classifier.
2. **Support Vector Machine (SVM):** Effective in high-dimensional spaces.
3. **Random Forest:** An ensemble method using multiple decision trees.
4. **CatBoost:** Gradient boosting algorithm optimized for categorical features.
5. **XGBoost:** Extreme Gradient Boosting algorithm efficient for large datasets.
6. **LightGBM:** A gradient boosting framework that uses tree-based learning algorithms.
7. **K-Means Clustering:** Applied as an unsupervised learning approach (evaluated using Adjusted Rand Index).

### Deep Learning Models

1. **Embedding & Global Max Pooling:** Simple neural network with embedding layer.
2. **Long Short-Term Memory (LSTM):** Captures long-term dependencies in sequences.
3. **Gated Recurrent Unit (GRU):** Similar to LSTM but with a simpler architecture.
4. **Bidirectional GRU:** Processes sequences in both forward and backward directions.
5. **Convolutional Neural Network (CNN) 1D:** Extracts local features from sequences.
6. **Convolutional Neural Network (CNN) 2D:** Adapted for text by representing sequences as matrices.
7. **Recurrent Neural Network (RNN):** Processes sequences sequentially but may struggle with long-term dependencies.
8. **Bidirectional LSTM:** Enhances context understanding by processing sequences in both directions.

## Evaluation Metrics

For classification tasks, the following metrics were used:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Area Under the ROC Curve (ROC AUC)**
- **Area Under the Precision-Recall Curve (PR AUC)**

For clustering (K-Means):

- **Adjusted Rand Index (ARI)**

## Results

### Machine Learning Models

| Model                    | Feature          | Accuracy | Precision | Recall | F1-Score |
|--------------------------|------------------|----------|-----------|--------|----------|
| Logistic Regression      | TF-IDF           | High     | High      | High   | High     |
| Logistic Regression      | Count Vectorizer | High     | High      | High   | High     |
| Logistic Regression      | Word2Vec         | Moderate | Moderate  | Moderate| Moderate|
| SVM                      | TF-IDF           | High     | High      | High   | High     |
| SVM                      | Count Vectorizer | High     | High      | High   | High     |
| SVM                      | Word2Vec         | Moderate | Moderate  | Moderate| Moderate|
| Random Forest            | TF-IDF           | High     | High      | High   | High     |
| Random Forest            | Count Vectorizer | High     | High      | High   | High     |
| Random Forest            | Word2Vec         | Moderate | Moderate  | Moderate| Moderate|
| CatBoost                 | TF-IDF           | High     | High      | High   | High     |
| CatBoost                 | Count Vectorizer | High     | High      | High   | High     |
| CatBoost                 | Word2Vec         | Moderate | Moderate  | Moderate| Moderate|
| XGBoost                  | TF-IDF           | High     | High      | High   | High     |
| XGBoost                  | Count Vectorizer | High     | High      | High   | High     |
| XGBoost                  | Word2Vec         | Moderate | Moderate  | Moderate| Moderate|
| LightGBM                 | TF-IDF           | High     | High      | High   | High     |
| LightGBM                 | Count Vectorizer | High     | High      | High   | High     |
| LightGBM                 | Word2Vec         | Moderate | Moderate  | Moderate| Moderate|
| K-Means                  | TF-IDF           | N/A      | N/A       | N/A    | ARI Score|
| K-Means                  | Count Vectorizer | N/A      | N/A       | N/A    | ARI Score|

### Deep Learning Models

| Model                   | Accuracy | Precision | Recall | F1-Score | ROC AUC | PR AUC |
|-------------------------|----------|-----------|--------|----------|---------|--------|
| Embedding & Max Pooling | Moderate | Moderate  | Moderate| Moderate| Moderate| Moderate|
| LSTM                    | High     | High      | High   | High     | High    | High   |
| GRU                     | High     | High      | High   | High     | High    | High   |
| Bidirectional GRU       | High     | High      | High   | High     | High    | High   |
| CNN 1D                  | High     | High      | High   | High     | High    | High   |
| CNN 2D                  | Moderate | Moderate  | Moderate| Moderate| Moderate| Moderate|
| Simple RNN              | Moderate | Moderate  | Moderate| Moderate| Moderate| Moderate|
| Bidirectional LSTM      | **Highest** | **Highest** | **Highest** | **Highest** | **Highest** | **Highest** |

**Note:** The Bidirectional LSTM achieved the best performance among deep learning models, indicating its effectiveness in capturing contextual information in both directions.

## Conclusion

- **Feature Representation:** TF-IDF and Count Vectorizer outperformed Word2Vec embeddings in traditional machine learning models.
- **Best Performing Models:**
  - **Machine Learning:** Logistic Regression, SVM, Random Forest, CatBoost, XGBoost, and LightGBM showed comparable high performance with TF-IDF and Count Vectorizer features.
  - **Deep Learning:** Bidirectional LSTM achieved the best results, capturing complex patterns in text data.
- **Neural Networks vs. Traditional Models:** Deep learning models, particularly those with recurrent architectures, demonstrated superior ability in handling sequence data for cyberbullying detection.

We however would like to put ourselves in a real world situation and envision working with business stakeholders. We think that LGBM may be the most practical solution as it is easier to explain and achieve similar results as Bidirectional LSTM. Although Bidirectional LSTM edges out LGBM in terms of correctly classifying true negative instances, it is important to remember the context of our project which is to identify cyberbullying cases. Both LGBM and Bidirectional LSTM actually achieved similar results. But in an academic setting, for pure optimisation, Bidirectional LSTM is a more attractive model to use as we have more ways to optimise it, and there is the added incentive of it being the most performant out of all out the box.

## Dependencies

The project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- imblearn
- nltk
- gensim
- tensorflow
- keras
- catboost
- lightgbm
- xgboost
- wordcloud

## How to Run the Code

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/cyberbullying-detection.git
   cd cyberbullying-detection
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.7 or higher installed.

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Necessary NLTK Data:**

   In your Python environment:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Run the Script:**

   ```bash
   python cyberbullying_detection.py
   ```

5. **Results:**

   - The script will output performance metrics for each model.
   - Plots for ROC and PR curves will be displayed.
   - Models and embeddings are saved for future use.

## References

- **Dataset:** [Cyberbullying Tweets Dataset](https://www.kaggle.com/datasets/)

- **Word Embeddings:**
  - [Word2Vec](https://arxiv.org/abs/1301.3781)
  - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

- **Machine Learning Algorithms:**
  - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
  - [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - [CatBoost](https://catboost.ai/)
  - [XGBoost](https://xgboost.readthedocs.io/en/latest/)
  - [LightGBM](https://lightgbm.readthedocs.io/en/latest/)

- **Deep Learning Models:**
  - [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network)
  - [Long Short-Term Memory Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - [Gated Recurrent Units](https://arxiv.org/abs/1406.1078)
  - [Convolutional Neural Networks](https://cs.nyu.edu/~yann/2010f-G22-2565-001/docs/lecun-98.pdf)
  - [Bidirectional RNNs](https://ieeexplore.ieee.org/document/650093)

---

**Disclaimer:** This project is for educational purposes only. The dataset and models are used to demonstrate machine learning and deep learning techniques for text classification. Always ensure ethical considerations when working with sensitive data.