import fitz  # PyMuPDF for PDF extraction
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tkinter import Tk, filedialog

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# ---------------------- File Handling Functions ----------------------

def extract_text_with_pymupdf(file_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def read_text_file(file_path):
    """Reads content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_file_path():
    """Opens file dialog for the user to select a file."""
    Tk().withdraw()  # Hides the main window
    file_path = filedialog.askopenfilename(title="Select a file", 
                                           filetypes=[("PDF files", "*.pdf"), ("Text files", "*.txt")])
    return file_path

# ---------------------- B+ Tree Implementation ----------------------

class Node:
    """Represents a node in the B+ Tree."""
    def __init__(self, order):
        self.order = order
        self.values = []
        self.keys = []
        self.nextKey = None
        self.parent = None
        self.check_leaf = False

class BplusTree:
    """Implements a B+ Tree with insertion and deletion operations."""
    def __init__(self, order):
        self.root = Node(order)
        self.root.check_leaf = True

    def search(self, value):
        """Searches for the correct leaf node where the value should be inserted."""
        current_node = self.root
        while not current_node.check_leaf:
            temp_values = current_node.values
            i = 0
            while i < len(temp_values) and value > temp_values[i]:
                i += 1
            
            # Check if i is within bounds of keys
            if i >= len(current_node.keys):  # Prevent IndexError
                return current_node
            
            current_node = current_node.keys[i]
        
        return current_node

    def insert(self, value):
        """Inserts a value into the B+ Tree."""
        current_node = self.search(value)
        current_node.values.append(value)
        current_node.values.sort()

        # If node overflows, split it
        if len(current_node.values) >= current_node.order:
            self.split(current_node)

    def split(self, node):
        """Splits a node when overflow occurs."""
        mid_index = len(node.values) // 2
        mid_value = node.values[mid_index]

        new_node = Node(node.order)
        new_node.check_leaf = node.check_leaf
        new_node.values = node.values[mid_index:]
        node.values = node.values[:mid_index]

        if node.check_leaf:
            new_node.nextKey = node.nextKey
            node.nextKey = new_node

        if not node.check_leaf:
            new_node.keys = node.keys[mid_index + 1:]
            node.keys = node.keys[:mid_index + 1]

        if node == self.root:
            new_root = Node(node.order)
            new_root.values = [mid_value]
            new_root.keys = [node, new_node]
            self.root = new_root
            node.parent = self.root
            new_node.parent = self.root
        else:
            parent = node.parent
            index = parent.keys.index(node)
            parent.values.insert(index, mid_value)
            parent.keys.insert(index + 1, new_node)
            new_node.parent = parent

            if len(parent.values) >= parent.order:
                self.split(parent)

    def delete(self, value):
        """Deletes a value from the B+ Tree."""
        current_node = self.search(value)

        if value in current_node.values:
            current_node.values.remove(value)

            # Handle underflow
            if len(current_node.values) < (current_node.order - 1) // 2:
                self.handle_underflow(current_node)

    def handle_underflow(self, node):
        """Handles node underflow by redistributing or merging nodes."""
        if node == self.root:
            if len(node.values) == 0 and not node.check_leaf:
                self.root = node.keys[0]
                self.root.parent = None
            return

        parent = node.parent
        index = parent.keys.index(node)

        left_sibling = parent.keys[index - 1] if index > 0 else None
        right_sibling = parent.keys[index + 1] if index + 1 < len(parent.keys) else None

        # Try redistribution
        if left_sibling and len(left_sibling.values) > (left_sibling.order - 1) // 2:
            borrowed_value = left_sibling.values.pop(-1)
            node.values.insert(0, parent.values[index - 1])
            parent.values[index - 1] = borrowed_value
            return

        if right_sibling and len(right_sibling.values) > (right_sibling.order - 1) // 2:
            borrowed_value = right_sibling.values.pop(0)
            node.values.append(parent.values[index])
            parent.values[index] = borrowed_value
            return

        # Merge nodes
        if left_sibling:
            left_sibling.values.extend([parent.values.pop(index - 1)] + node.values)
            left_sibling.nextKey = node.nextKey
            parent.keys.pop(index)
        elif right_sibling:
            node.values.extend([parent.values.pop(index)] + right_sibling.values)
            node.nextKey = right_sibling.nextKey
            parent.keys.pop(index + 1)

        if len(parent.values) < (parent.order - 1) // 2:
            self.handle_underflow(parent)

# ---------------------- Execution Code ----------------------

# Automatically get file path
file_path = get_file_path()

# Process selected file
if file_path.endswith(".pdf"):
    extracted_text = extract_text_with_pymupdf(file_path)
    documents = [extracted_text]
elif file_path.endswith(".txt"):
    file_contents = read_text_file(file_path)
    documents = [file_contents]
else:
    print("Unsupported file type. Please select a PDF or TXT file.")
    exit()

# Vectorization (CountVectorizer & TF-IDF)
count_vectorizer = CountVectorizer(stop_words='english')
X_count = count_vectorizer.fit_transform(documents)

print("\nFeature Names (CountVectorizer):", count_vectorizer.get_feature_names_out())
print("Vectorized Output (CountVectorizer):\n", X_count.toarray())

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(documents)

print("\nFeature Names (TF-IDF):", tfidf_vectorizer.get_feature_names_out())
print("Vectorized Output (TF-IDF):\n", X_tfidf.toarray())

# ---------------------- Insert into B+ Tree ----------------------

# Initialize B+ Tree
order = 4
bplustree = BplusTree(order)

# Insert words into the B+ Tree
for word in count_vectorizer.get_feature_names_out():
    bplustree.insert(word)

print("\nB+ Tree after insertions:")
def printTree(tree):
    lst = [tree.root]
    while lst:
        next_level = []
        for node in lst:
            print(node.values, end=" | ")
            if not node.check_leaf:
                next_level.extend(node.keys)
        print()
        lst = next_level

printTree(bplustree)

# ---------------------- Data Preprocessing Functions ----------------------

def preprocess_text(text):
    """Preprocess the text by normalizing, tokenizing, and removing punctuation."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ---------------------- Similarity Calculation ----------------------

def get_bert_embeddings(text):
    """Generate BERT embeddings for the input text."""
    # Preprocess the text
    text = preprocess_text(text)
    
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get the embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings of the [CLS] token (first token)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def calculate_similarity(doc1, doc2):
    """Calculate cosine similarity between two documents using BERT embeddings."""
    # Get embeddings for both documents
    embedding1 = get_bert_embeddings(doc1)
    embedding2 = get_bert_embeddings(doc2)
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cosine_similarity[0][0]

# ---------------------- Contextual Analysis ----------------------

def extract_domain_specific_keywords(doc, domain_keywords):
    """Extract domain-specific keywords from the document."""
    # Preprocess document
    doc = preprocess_text(doc)
    # Tokenize the document
    words = doc.split()
    # Filter for domain-specific keywords
    specific_keywords = [word for word in words if word in domain_keywords]
    return specific_keywords

# ---------------------- Additional Contextual Analysis ----------------------

def identify_frequent_terms(doc, top_n=5):
    """Identify the most frequent terms in the document."""
    # Preprocess document
    doc = preprocess_text(doc)
    words = doc.split()
    word_freq = {}
    
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    return sorted_words[:top_n]

# ---------------------- Example Usage ----------------------

# Define domain-specific keywords
domain_keywords = ['security', 'data', 'analysis', 'algorithm', 'similarity', 'document', 'text']

# Assuming 'extracted_text' is the text from the PDF or TXT file
doc1 = extracted_text  # First document
doc2 = "This is a sample document for testing similarity in data analysis."  # Second document

# Calculate BERT similarity
similarity_score = calculate_similarity(doc1, doc2)
print(f"BERT Cosine Similarity Score: {similarity_score:.4f}")

# Extract domain-specific keywords
keywords = extract_domain_specific_keywords(doc1, domain_keywords)
print(f"Domain-Specific Keywords: {keywords}")

# Identify frequent terms in the first document
frequent_terms = identify_frequent_terms(doc1, top_n=10)
print(f"Most Frequent Terms in Document 1: {frequent_terms}")

# If you have multiple documents, you can loop through them
documents = [doc1, doc2, "Another document for testing purposes."]
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        score = calculate_similarity(documents[i], documents[j])
        print(f"BERT Cosine Similarity between Document {i+1} and Document {j+1}: {score:.4f}")
