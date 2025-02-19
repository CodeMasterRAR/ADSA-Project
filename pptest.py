import fitz  # PyMuPDF

def extract_text_with_pymupdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

# Define the path to your PDF file
pdf_file_path = "C:/Users/91900/Desktop/ADSA PROJEECT/testfile.pdf"

# Extract text
extracted_text = extract_text_with_pymupdf(pdf_file_path)
#print("Extracted Text:\n", extracted_text)

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
#print("File Contents:\n", file_contents)


ans=input("enter type of file:")
if ans=='textfile':
    documents = [file_contents]
if ans=='pdf':
    documents=[extracted_text]

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