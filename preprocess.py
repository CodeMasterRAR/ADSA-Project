import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample documents
documents = ["This is a sample sentence.", "This is another example."]

# Vectorization with stop words removal
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

print("Feature Names:", vectorizer.get_feature_names_out())
print("Vectorized Output (CountVectorizer):\n", X.toarray())

# TfidfVectorizer with stop words removal
vectorizer = TfidfVectorizer(stop_words='english')  # Added stop_words for consistency
X = vectorizer.fit_transform(documents)  

print("\nFeature Names (TF-IDF):", vectorizer.get_feature_names_out())
print("Vectorized Output (TF-IDF):\n", X.toarray())
