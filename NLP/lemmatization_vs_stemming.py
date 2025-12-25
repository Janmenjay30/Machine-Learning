import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def main():
    download_nltk_resources()
    
    # Initialize Stemmer and Lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Load Data
    try:
        df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])
    except FileNotFoundError:
        print("Dataset not found.")
        return

    # Let's take a small sample to demonstrate the difference clearly
    sample_messages = df['message'].iloc[:5].tolist()
    # Add a custom sentence to show irregular verbs which is where Lemmatization shines
    sample_messages.append("I went running and studies historically.")

    print(f"{'Original':<50} | {'Stemming':<50} | {'Lemmatization':<50}")
    print("-" * 155)

    corpus_lemmatized = []
    
    for i, sentence in enumerate(sample_messages):
        # Cleaning
        sentence_clean = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence_clean = sentence_clean.lower()
        words = sentence_clean.split()
        
        # Comparison logic
        stemmed_words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        
        # Store for BoW
        corpus_lemmatized.append(' '.join(lemmatized_words))

        # Print comparison for the first few
        orig_str = " ".join(words)
        stem_str = " ".join(stemmed_words)
        lemm_str = " ".join(lemmatized_words)
        
        print(f"{orig_str[:48]:<50} | {stem_str[:48]:<50} | {lemm_str[:48]:<50}")
    
    print("\n" + "="*50)
    print("Why Lemmatization is often 'Better':")
    print("Look at the last sentence: 'studies historically'")
    print(f"Stemmer got: {[stemmer.stem('studies'), stemmer.stem('historically')]}")
    print(f"Lemmatizer got: {[lemmatizer.lemmatize('studies', pos='v'), lemmatizer.lemmatize('historically')]}") 
    print("(Note: Lemmatizer often needs POS tags to be perfect, e.g., 'studies' -> 'study' needs pos='v')")

    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=10) # Small for demo
    X = cv.fit_transform(corpus_lemmatized)
    
    print("\nBag of Words on Lemmatized Data (Top 10 features):")
    print(cv.get_feature_names_out())
    print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()))

if __name__ == "__main__":
    main()
