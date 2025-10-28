# E-commerce Sentiment Analysis

## 📋 About the Project

This project implements a **Machine Learning** solution to automate customer feedback classification on an e-commerce platform. The goal is to automatically classify product reviews as **positive** or **negative**, enabling the company to:

- Reduce costs and time spent on manual feedback analysis
- Quickly identify products with issues or improvement opportunities
- Prioritize customer support for negative reviews, improving overall customer experience

## 🎯 Objective

Build a Machine Learning model capable of automatically classifying product reviews as:
- **Positive** (1)
- **Negative** (0)

Reviews with **neutral** sentiment are removed from the final dataset for a more focused binary classification model.

## 📊 Dataset

The dataset used (`sentiment.csv`) contains:
- **171,380 product review records**
- **6 columns**: `ProductName`, `ProductPrice`, `Rate`, `Review`, `Summary`, `Sentiment`
- Distribution: Positive, negative, and neutral reviews

After cleaning and removing null values and neutral reviews, the final dataset for modeling contains **157,241 records**.

## 🔧 Technologies Used

- **Python 3.11**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine Learning
  - `TfidfVectorizer` - Text vectorization
  - `LogisticRegression` - Classification model
  - `Pipeline` - Processing pipeline
  - `train_test_split` - Data splitting
- **Seaborn & Matplotlib** - Data visualization
- **Unicodedata & Regex** - Text cleaning

## 📁 Project Structure

```
DS-Spark/
├── e-commerce.ipynb        # Main notebook with complete pipeline
├── sentiment.csv            # Original dataset
├── sentiment_clean.csv      # Cleaned dataset
└── README.md               # This file
```

## 🔄 Processing Pipeline

The project follows a structured Data Science methodology:

### 1. Business Problem Definition
- Need identification: automate feedback analysis
- Definition of objectives and success metrics

### 2. Library Imports
- Installation and import of all necessary dependencies

### 3. Data Loading and Understanding
- Initial dataset analysis
- Structure, types, and missing values verification

### 4. Exploratory Data Analysis (EDA)
- Sentiment distribution
- Text patterns
- Descriptive statistics

### 5. Data Cleaning
- **Missing value removal**
- **Sentiment label normalization** (lowercase, trim)
- **Numeric column conversion** (`ProductPrice`, `Rate`)
- **Text cleaning**:
  - Unicode normalization (accent removal)
  - Lowercase conversion
  - Punctuation and number removal
  - Extra space removal

### 6. Feature Engineering
- Creation of `Clean_Review` column with processed text
- Creation of `Clean_Summary` column
- Creation of `sentiment_label` (mapping: positive=1, negative=0)
- Creation of text length metrics
- Removal of neutral records

### 7. Predictive Modeling
- Stratified data split (75% train, 25% test)
- Machine Learning Pipeline:
  1. **TF-IDF Vectorization** - Text to numerical vectors transformation
  2. **StandardScaler** - Normalization (without centering, appropriate for sparse data)
  3. **Logistic Regression** - Binary classifier

## 🧹 Text Cleaning Function

The `dse_cleaning_text()` function performs complete text cleaning:

```python
def dse_cleaning_text(text):
    # 1. Normalizes and removes accents (NFKD)
    # 2. Converts to lowercase
    # 3. Removes punctuation, numbers, and special characters
    # 4. Removes extra spaces
```

## 📈 Data Split

- **Train**: 75% of data
- **Test**: 25% of data
- **Stratified Split**: Maintains class proportion in both sets
- **Random State**: 42 (for reproducibility)

## 🚀 Next Steps

The notebook is structured to include:
- Model training
- Metric evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix
- Result visualizations
- Trained model saving (using `joblib`)

## 📝 Notes

- The dataset uses `latin-1` encoding
- Portuguese stop words are removed during TF-IDF vectorization
- The model uses `solver='liblinear'` for binary classification problems

## 👤 Author

**Eduardo Data Science**

---

*Project developed to demonstrate a complete Machine Learning pipeline for sentiment analysis in e-commerce texts.*
