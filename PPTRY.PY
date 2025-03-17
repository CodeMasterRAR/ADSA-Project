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
import ast
from typing import List, Tuple
import astor  # You'll need to pip install astor

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

def read_python_file(file_path):
    """Reads content from a Python (.py) file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_file_paths():
    """
    Opens file dialog for the user to select one or more PDF, text, or Python files.
    This function returns a list of file paths.
    """
    Tk().withdraw()  # Hides the main window
    file_paths = filedialog.askopenfilenames(
        title="Select PDF, Text, or Python Files",
        filetypes=[
            ("All Supported Files", "*.pdf *.txt *.py"),
            ("PDF files", "*.pdf"),
            ("Text files", "*.txt"),
            ("Python files", "*.py")
        ],
        initialdir="."  # Start in current directory
    )
    return list(file_paths)

def get_python_file_paths():
    """Opens file dialog for the user to select exactly two Python files."""
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Two Python Files to Compare",
        filetypes=[("Python files", "*.py")],
        initialdir="."
    )
    return list(file_paths)

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

# Get multiple file paths (PDFs, TXT, or PY files)
file_paths = get_file_paths()

if not file_paths:
    print("No files selected. Exiting.")
    exit()

# ---------------------- Process Selected Files ----------------------

# Process selected files and collect their content
documents = []
file_names = []
for file_path in file_paths:
    if file_path.lower().endswith(".pdf"):
        text = extract_text_with_pymupdf(file_path)
        documents.append(text)
        file_names.append(file_path.split('/')[-1])  # Store the file name
    elif file_path.lower().endswith(".txt"):
        text = read_text_file(file_path)
        documents.append(text)
        file_names.append(file_path.split('/')[-1])  # Store the file name
    elif file_path.lower().endswith(".py"):
        text = read_python_file(file_path)
        documents.append(text)
        file_names.append(file_path.split('/')[-1])  # Store the file name
    else:
        print(f"Unsupported file type for file {file_path}. Skipping.")

# Print the documents with their names
print("\nUploaded Documents:")
for name, content in zip(file_names, documents):
    print(f"\nDocument Name: {name}\nContent:\n{content}\n")

# ---------------------- Vectorization (CountVectorizer & TF-IDF) ----------------------

# Combine all documents for vectorization
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

# Insert words from the CountVectorizer vocabulary into the B+ Tree
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

def identify_frequent_terms(doc, top_n=5):
    """Identify the most frequent terms in the document."""
    # Preprocess document
    doc = preprocess_text(doc)
    words = doc.split()
    word_freq = {}
    
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    return sorted_words[:top_n]

# ---------------------- Example Usage ----------------------

# Define domain-specific keywords
domain_keywords = ['security', 'data', 'analysis', 'algorithm', 'similarity', 'document', 'text']

# ---------------------- Similarity Calculations ----------------------

# If you uploaded multiple documents, calculate pairwise BERT similarity:
if len(documents) > 1:
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            score = calculate_similarity(documents[i], documents[j])
            print(f"BERT Cosine Similarity between '{file_names[i]}' and '{file_names[j]}': {score:.4f}")
else:
    # If only one document was uploaded, compare it to a sample document
    doc1 = documents[0]
    doc2 = "C:/Users/91900/Downloads/ADSA BERT PDFTXT/SUBWAY.txt"
    similarity_score = calculate_similarity(doc1, doc2)
    print(f"BERT Cosine Similarity Score: {similarity_score:.4f}")

# ---------------------- Additional Contextual Analysis ----------------------

# For example, analyze the first document
doc = documents[0]
keywords = extract_domain_specific_keywords(doc, domain_keywords)
frequent_terms = identify_frequent_terms(doc, top_n=10)
print(f"Domain-Specific Keywords in Document '{file_names[0]}': {keywords}")
print(f"Most Frequent Terms in Document '{file_names[0]}': {frequent_terms}")

# ---------------------- AST Comparison ----------------------

def normalize_ast(node):
    """Normalize an AST by standardizing all variable names, literals, and function names."""
    if isinstance(node, ast.Name):
        # Replace all variable names with 'VAR'
        return ast.Name(id='VAR', ctx=node.ctx)
    
    elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
        # Replace all literals with a standard placeholder
        return ast.Constant(value='LITERAL')
    
    elif isinstance(node, ast.FunctionDef):
        # Normalize function name and body
        return ast.FunctionDef(
            name='FUNC',
            args=normalize_ast(node.args),
            body=[normalize_ast(n) for n in node.body],
            decorator_list=[],
            returns=None
        )
    
    elif isinstance(node, ast.arguments):
        # Normalize function arguments
        return ast.arguments(
            posonlyargs=[normalize_ast(arg) for arg in node.posonlyargs],
            args=[normalize_ast(arg) for arg in node.args],
            kwonlyargs=[normalize_ast(arg) for arg in node.kwonlyargs],
            kw_defaults=[normalize_ast(default) if default else None for default in node.kw_defaults],
            defaults=[normalize_ast(default) if default else None for default in node.defaults]
        )
    
    elif isinstance(node, ast.arg):
        # Normalize argument names
        return ast.arg(arg='ARG', annotation=None)
    
    elif isinstance(node, ast.Call):
        # Normalize function calls
        return ast.Call(
            func=normalize_ast(node.func),
            args=[normalize_ast(arg) for arg in node.args],
            keywords=[normalize_ast(kw) for kw in node.keywords]
        )
    
    elif isinstance(node, ast.Assign):
        # Normalize assignments
        return ast.Assign(
            targets=[normalize_ast(target) for target in node.targets],
            value=normalize_ast(node.value)
        )
    
    elif isinstance(node, ast.BinOp):
        # Preserve operation structure but normalize operands
        return ast.BinOp(
            left=normalize_ast(node.left),
            op=node.op,
            right=normalize_ast(node.right)
        )
    
    elif isinstance(node, ast.Expr):
        # Normalize expressions
        return ast.Expr(value=normalize_ast(node.value))
    
    elif isinstance(node, list):
        # Handle lists of nodes
        return [normalize_ast(n) for n in node]
    
    # For any other nodes, recursively normalize their fields
    elif isinstance(node, ast.AST):
        return type(node)(**{
            field: normalize_ast(value) if isinstance(value, (ast.AST, list)) else value
            for field, value in ast.iter_fields(node)
        })
    
    return node

def compare_asts(file1_path: str, file2_path: str) -> Tuple[float, List[str]]:
    """
    Compare two Python files using AST analysis.
    Returns similarity score and list of similar code structures found.
    """
    try:
        # Parse both files into ASTs
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            ast1 = ast.parse(f1.read())
            ast2 = ast.parse(f2.read())

        # Normalize both ASTs
        normalized_ast1 = ast.fix_missing_locations(normalize_ast(ast1))
        normalized_ast2 = ast.fix_missing_locations(normalize_ast(ast2))

        # Convert normalized ASTs to string representation for comparison
        ast1_code = astor.to_source(normalized_ast1)
        ast2_code = astor.to_source(normalized_ast2)

        # Calculate similarity score
        import difflib
        similarity = difflib.SequenceMatcher(None, ast1_code, ast2_code).ratio()

        # Find similar structures
        similar_structures = []
        for node1 in ast.walk(ast1):
            if isinstance(node1, (ast.FunctionDef, ast.ClassDef)):
                for node2 in ast.walk(ast2):
                    if isinstance(node2, type(node1)):
                        # Compare normalized versions of these nodes
                        norm1 = astor.to_source(normalize_ast(node1))
                        norm2 = astor.to_source(normalize_ast(node2))
                        if norm1 == norm2:
                            similar_structures.append(f"Similar {type(node1).__name__}: {node1.name} ↔ {node2.name}")

        # Adjust similarity threshold for identical logic
        if len(similar_structures) > 0 and similarity > 0.9:
            similarity = 1.0  # If structure is identical, consider it 100% similar

        return similarity, similar_structures

    except Exception as e:
        print(f"Error during AST comparison: {e}")
        return 0.0, []

# Modify the main execution code to focus on Python files
if __name__ == "__main__":
    # Get exactly two Python files
    file_paths = get_python_file_paths()
    
    if len(file_paths) != 2:
        print("Please select exactly two Python files for comparison.")
        exit()

    # Perform AST comparison
    similarity_score, similar_structures = compare_asts(file_paths[0], file_paths[1])
    
    print(f"\nAST Comparison Results:")
    print(f"File 1: {file_paths[0]}")
    print(f"File 2: {file_paths[1]}")
    print(f"Structural Similarity Score: {similarity_score:.2%}")
    
    if similar_structures:
        print("\nSimilar Code Structures Found:")
        for structure in similar_structures:
            print(f"- {structure}")
    
    # Provide interpretation
    if similarity_score > 0.8:
        print("\nWARNING: High similarity detected! The code structure is very similar and might be a case of code copying.")
    elif similarity_score > 0.6:
        print("\nNOTICE: Moderate similarity detected. Some code structures are similar.")
    else:
        print("\nLow similarity detected. The code structures appear to be different.")