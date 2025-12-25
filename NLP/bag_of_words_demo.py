import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def main():
    # Dataset: List of sentences
    documents = [
        "I love deep learning",
        "I love machine learning",
        "I enjoy learning deep learning concepts",
        "Machine learning is fascinating",
        "Deep learning is a subset of machine learning"
    ]

    print("Dataset (Documents):")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc}")
    print("-" * 30)

    # Create the Bag of Words model
    # CountVectorizer converts a collection of text documents to a matrix of token counts
    vectorizer = CountVectorizer()
    
    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)

    # Convert to DataFrame for better visualization
    # The columns are the unique words (vocabulary) found in the documents
    df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    print("\nBag of Words Representation:")
    print(df_bow)

    # Optional: Show vocabulary
    print("\nVocabulary Mapping:")
    print(vectorizer.vocabulary_)

if __name__ == "__main__":
    main()
