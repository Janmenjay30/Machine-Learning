import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def main():
    print("Loading SMS Spam Collection dataset...")
    
    # Load dataset
    # The dataset is tab-separated (TSV) and has no header
    try:
        df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
    except FileNotFoundError:
        print("Error: 'SMSSpamCollection' file not found. Please ensure it is in the same directory.")
        return

    print(f"Dataset loaded successfully due to {len(df)} records.")
    print("\nFirst 5 rows:")
    print(df.head())

    # Text Preprocessing (Basic)
    # We will use the 'message' column
    corpus = df['message']

    # Initialize Bag of Words
    print("\nCreating Bag of Words model...")
    vectorizer = CountVectorizer(max_features=2500) # Limit to top 2500 frequent words to avoid huge sparsity
    X = vectorizer.fit_transform(corpus)

    # Convert to array (for visualization purposes, usually we keep it sparse for ML)
    # Using a small subset for display or the memory might explode with 5000+ rows and 8000+ words
    print("Shape of Bag of Words Matrix:", X.shape)
    
    # Show feature names (a few of them)
    print("\nFirst 20 features (words in vocabulary):")
    print(vectorizer.get_feature_names_out()[:20])
    
    print("\nSample of the Bag of Words Matrix (first 5 rows, first 10 columns):")
    # We can densify just a small chunk
    print(pd.DataFrame(X[:5, :10].toarray(), columns=vectorizer.get_feature_names_out()[:10]))

if __name__ == "__main__":
    main()
