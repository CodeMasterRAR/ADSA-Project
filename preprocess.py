import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Define the path to your text file
file_path = "C:/Users/91900/Desktop/ADSA PROJEECT/sampletext.txt"

# Read the contents of the file
file_contents = read_text_file(file_path)

# Print the contents
print("File Contents:\n", file_contents)

# Wrap the contents in a list for vectorization
documents = [file_contents]

# Vectorization with stop words removal using CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_count = count_vectorizer.fit_transform(documents)

print("\nFeature Names (CountVectorizer):", count_vectorizer.get_feature_names_out())
print("Vectorized Output (CountVectorizer):\n", X_count.toarray())

# TfidfVectorizer with stop words removal
tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # Added stop_words for consistency
X_tfidf = tfidf_vectorizer.fit_transform(documents)

print("\nFeature Names (TF-IDF):", tfidf_vectorizer.get_feature_names_out())
print("Vectorized Output (TF-IDF):\n", X_tfidf.toarray())