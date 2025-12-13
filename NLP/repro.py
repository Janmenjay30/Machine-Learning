import nltk
print("Imported nltk successfully")
try:
    nltk.download('stopwords')
    print("Downloaded stopwords successfully")
except Exception as e:
    print(f"Error: {e}")
