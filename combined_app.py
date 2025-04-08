from flask import Flask, render_template, request, jsonify
import ast
import fitz  # PyMuPDF for PDF extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import astor  # For AST to source conversion
import difflib

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Initialize Flask app
app = Flask(__name__)

# ---------------------- Normalization Function ----------------------
def normalize(x, min_val, max_val):
    if max_val - min_val == 0:
        return 1
    return (x - min_val) / (max_val - min_val)

# ---------------------- B+ Tree Implementation ----------------------
class Node:
    def __init__(self, order):
        self.order = order
        self.values = []
        self.keys = []
        self.nextKey = None
        self.parent = None
        self.check_leaf = False

class BplusTree:
    def __init__(self, order):
        self.root = Node(order)
        self.root.check_leaf = True

    def search(self, value):
        current_node = self.root
        while not current_node.check_leaf:
            temp_values = current_node.values
            i = 0
            while i < len(temp_values) and value > temp_values[i]:
                i += 1
            if i >= len(current_node.keys):
                return current_node
            current_node = current_node.keys[i]
        return current_node

    def insert(self, value):
        current_node = self.search(value)
        current_node.values.append(value)
        current_node.values.sort()
        if len(current_node.values) >= current_node.order:
            self.split(current_node)

    def split(self, node):
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

# ---------------------- AST Normalization and Comparison ----------------------
def normalize_ast(node):
    if isinstance(node, ast.Name):
        return ast.Name(id='VAR', ctx=node.ctx)
    elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
        return ast.Constant(value='LITERAL')
    elif isinstance(node, ast.FunctionDef):
        return ast.FunctionDef(
            name='FUNC',
            args=normalize_ast(node.args),
            body=[normalize_ast(n) for n in node.body],
            decorator_list=[],
            returns=None
        )
    elif isinstance(node, ast.arguments):
        return ast.arguments(
            posonlyargs=[normalize_ast(arg) for arg in node.posonlyargs],
            args=[normalize_ast(arg) for arg in node.args],
            kwonlyargs=[normalize_ast(arg) for arg in node.kwonlyargs],
            kw_defaults=[normalize_ast(default) if default else None for default in node.kw_defaults],
            defaults=[normalize_ast(default) if default else None for default in node.defaults]
        )
    elif isinstance(node, ast.arg):
        return ast.arg(arg='ARG', annotation=None)
    elif isinstance(node, ast.Call):
        return ast.Call(
            func=normalize_ast(node.func),
            args=[normalize_ast(arg) for arg in node.args],
            keywords=[normalize_ast(kw) for kw in node.keywords]
        )
    elif isinstance(node, ast.Assign):
        return ast.Assign(
            targets=[normalize_ast(target) for target in node.targets],
            value=normalize_ast(node.value)
        )
    elif isinstance(node, ast.BinOp):
        return ast.BinOp(
            left=normalize_ast(node.left),
            op=node.op,
            right=normalize_ast(node.right)
        )
    elif isinstance(node, ast.Expr):
        return ast.Expr(value=normalize_ast(node.value))
    elif isinstance(node, list):
        return [normalize_ast(n) for n in node]
    elif isinstance(node, ast.AST):
        return type(node)(**{
            field: normalize_ast(value) if isinstance(value, (ast.AST, list)) else value
            for field, value in ast.iter_fields(node)
        })
    return node

def compare_asts(code1, code2):
    try:
        ast1 = ast.parse(code1)
        ast2 = ast.parse(code2)
        normalized_ast1 = ast.fix_missing_locations(normalize_ast(ast1))
        normalized_ast2 = ast.fix_missing_locations(normalize_ast(ast2))
        ast1_code = astor.to_source(normalized_ast1)
        ast2_code = astor.to_source(normalized_ast2)
        similarity = difflib.SequenceMatcher(None, ast1_code, ast2_code).ratio()
        similar_structures = []
        for node1 in ast.walk(ast1):
            if isinstance(node1, (ast.FunctionDef, ast.ClassDef)):
                for node2 in ast.walk(ast2):
                    if isinstance(node2, type(node1)):
                        norm1 = astor.to_source(normalize_ast(node1))
                        norm2 = astor.to_source(normalize_ast(node2))
                        if norm1 == norm2:
                            similar_structures.append(f"Similar {type(node1).__name__}: {node1.name} ↔ {node2.name}")
        if len(similar_structures) > 0 and similarity > 0.9:
            similarity = 1.0
        return similarity, similar_structures
    except Exception as e:
        return 0.0, [f"Error during AST comparison: {e}"]

# ---------------------- BERT Embeddings and Similarity ----------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_bert_embeddings(text):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def calculate_bert_similarity(code1, code2):
    embedding1 = get_bert_embeddings(code1)
    embedding2 = get_bert_embeddings(code2)
    cosine_similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cosine_similarity[0][0]

# ---------------------- Flask Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare-files', methods=['POST'])
def compare_files():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if not file1 or not file2:
        return jsonify({"error": "Please upload two Python files."}), 400

    code1 = file1.read().decode('utf-8')
    code2 = file2.read().decode('utf-8')

    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
    except SyntaxError as e:
        return jsonify({"error": f"Syntax error in uploaded files: {str(e)}"}), 400

    # Structural similarity (from app.py)
    classes_in_code1, classes_in_code2 = 0, 0
    methods_in_code1, methods_in_code2 = 0, 0
    functions_in_code1, functions_in_code2 = 0, 0
    expressions_in_code1, expressions_in_code2 = 0, 0

    for node in tree1.body:
        if isinstance(node, ast.ClassDef):
            classes_in_code1 += 1
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    methods_in_code1 += 1
        elif isinstance(node, ast.FunctionDef):
            functions_in_code1 += 1
        elif isinstance(node, ast.Expr):
            expressions_in_code1 += 1

    for node in tree2.body:
        if isinstance(node, ast.ClassDef):
            classes_in_code2 += 1
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    methods_in_code2 += 1
        elif isinstance(node, ast.FunctionDef):
            functions_in_code2 += 1
        elif isinstance(node, ast.Expr):
            expressions_in_code2 += 1

    class_sim = 1 - abs(classes_in_code1 - classes_in_code2) * 0.5
    meth_sim = 1 - abs(methods_in_code1 - methods_in_code2) * 0.5
    exp_sim = 1 - abs(expressions_in_code1 - expressions_in_code2) * 0.5
    func_sim = 1 - abs(functions_in_code1 - functions_in_code2) * 0.5

    # Naming similarity (from app.py)
    same_class, same_method, same_func = 0, 0, 0
    for node1, node2 in zip(tree1.body, tree2.body):
        if isinstance(node1, ast.ClassDef) and isinstance(node2, ast.ClassDef):
            if node1.name == node2.name:
                same_class += 1
                for s_node1, s_node2 in zip(node1.body, node2.body):
                    if isinstance(s_node1, ast.FunctionDef) and isinstance(s_node2, ast.FunctionDef):
                        same_method += 1 if s_node1.name == s_node2.name else -1

    for node1 in tree1.body:
        if isinstance(node1, ast.FunctionDef):
            for node2 in tree2.body:
                if isinstance(node2, ast.FunctionDef) and node1.name == node2.name:
                    same_func += 1
                    break
            else:
                same_func -= 2.0

    max_functions = max(functions_in_code1, functions_in_code2)
    same_class_name = 1 if classes_in_code1 == 0 else normalize(same_class, -classes_in_code1, classes_in_code1)
    same_function_name = 1 if functions_in_code1 == 0 else normalize(same_func, -max_functions, max_functions)
    same_method_name = 1 if methods_in_code1 == 0 else normalize(same_method, -methods_in_code1, methods_in_code1)

    # Return similarity (from app.py)
    returns1 = [stmt.value for node in tree1.body if isinstance(node, ast.FunctionDef) for stmt in node.body if isinstance(stmt, ast.Return)]
    returns2 = [stmt.value for node in tree2.body if isinstance(node, ast.FunctionDef) for stmt in node.body if isinstance(stmt, ast.Return)]
    same_function_return = 1
    penalty = 0.35
    matched_indices = set()

    for r1 in returns1:
        match_found = False
        for i, r2 in enumerate(returns2):
            if i not in matched_indices and ast.dump(r1) == ast.dump(r2):
                match_found = True
                matched_indices.add(i)
                break
        if not match_found:
            same_function_return -= penalty

    # AST similarity (from pptry.py)
    ast_similarity, similar_structures = compare_asts(code1, code2)

    # BERT similarity (from pptry.py)
    bert_similarity = calculate_bert_similarity(code1, code2)

    # Dynamic weighting (extended from app.py)
    weighted_sum = 0
    total_weight = 0
    if classes_in_code1 > 0:
        weighted_sum += class_sim * 1 + same_class_name * 1
        total_weight += 2
    if methods_in_code1 > 0:
        weighted_sum += meth_sim * 1 + same_method_name * 1
        total_weight += 2
    if expressions_in_code1 > 0:
        weighted_sum += exp_sim * 1
        total_weight += 1
    if functions_in_code1 > 0:
        weighted_sum += func_sim * 3 + same_function_name * 2 + same_function_return * 1
        total_weight += 6
    weighted_sum += ast_similarity * 3 + bert_similarity * 2  # Additional weights for AST and BERT
    total_weight += 5

    if total_weight == 0:
        total_weight = 1
        weighted_sum = 1

    final_score = abs(weighted_sum / total_weight)
    similarity_percentage = "{:.2f}".format(final_score * 100)

    # B+ Tree for keywords (from pptry.py)
    bplustree = BplusTree(order=4)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform([code1, code2])
    for word in tfidf_vectorizer.get_feature_names_out():
        bplustree.insert(word)

    # Response
    result = {
        "result": similarity_percentage,
        "ast_similarity": "{:.2f}".format(ast_similarity * 100),
        "bert_similarity": "{:.2f}".format(bert_similarity * 100),
        "similar_structures": similar_structures,
        "details": {
            "class_sim": "{:.2f}".format(class_sim * 100),
            "meth_sim": "{:.2f}".format(meth_sim * 100),
            "exp_sim": "{:.2f}".format(exp_sim * 100),
            "func_sim": "{:.2f}".format(func_sim * 100),
            "same_class_name": "{:.2f}".format(same_class_name * 100),
            "same_method_name": "{:.2f}".format(same_method_name * 100),
            "same_function_name": "{:.2f}".format(same_function_name * 100),
            "same_function_return": "{:.2f}".format(same_function_return * 100)
        }
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)