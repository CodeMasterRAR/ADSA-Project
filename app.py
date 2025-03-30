from flask import Flask, render_template, request, jsonify
import ast


def normalize(x, min_val, max_val):
    if max_val - min_val == 0:
        return 1
    return (x - min_val) / (max_val - min_val)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare-files', methods=['POST'])
def pysimilarity():

    file1 = request.files.get('file1')
    file2 = request.files.get('file2')


    code1 = file1.read().decode('utf-8')
    code2 = file2.read().decode('utf-8')


    # Counters for basic structure
    classes_in_code1 = 0
    classes_in_code2 = 0

    methods_in_code1 = 0
    methods_in_code2 = 0

    functions_in_code1 = 0
    functions_in_code2 = 0

    expressions_in_code1 = 0
    expressions_in_code2 = 0

    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
    except SyntaxError as e:
        return jsonify({"error": f"Syntax error in uploaded files: {str(e)}"}), 400


    # Count elements in tree1
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

    # Count elements in tree2
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

    # ---------------- Structural similarity scores -----------------
    # Base scores start at 1 and then we subtract differences.
    class_sim = 1
    meth_sim = 1
    exp_sim = 1
    func_sim = 1

    # Increase penalties by using a multiplier (0.5)
    if functions_in_code1 != functions_in_code2:
        func_sim -= abs(functions_in_code1 - functions_in_code2) * 0.5

    if methods_in_code1 != methods_in_code2:
        meth_sim -= abs(methods_in_code1 - methods_in_code2) * 0.5

    if expressions_in_code1 != expressions_in_code2:
        exp_sim -= abs(expressions_in_code1 - expressions_in_code2) * 0.5

    if classes_in_code1 != classes_in_code2:
        class_sim -= abs(classes_in_code1 - classes_in_code2) * 0.5

    # ---------------- Naming similarity -----------------
    same_class = 0
    same_method = 0
    same_func = 0

    # Compare classes (if any)
    for node1, node2 in zip(tree1.body, tree2.body):
        if isinstance(node1, ast.ClassDef) and isinstance(node2, ast.ClassDef):
            if node1.name == node2.name:
                same_class += 1
                for s_node1, s_node2 in zip(node1.body, node2.body):
                    if isinstance(s_node1, ast.FunctionDef) and isinstance(s_node2, ast.FunctionDef):
                        same_method += 1
                    else:
                        same_method -= 1
            else:
                same_class -= 1

    # Compare functions using nested loops (ignoring names for return matching later)
    for node1 in tree1.body:
        if isinstance(node1, ast.FunctionDef):
            match_found = False
            for node2 in tree2.body:
                if isinstance(node2, ast.FunctionDef):
                    if node1.name == node2.name:
                        same_func += 1
                        match_found = True
                        break
            if not match_found:
                same_func -= 2.0  # Increased penalty for no match

    max_functions = max(functions_in_code1, functions_in_code2)
    same_class_name = 1 if classes_in_code1 == 0 else normalize(same_class, -classes_in_code1, classes_in_code1)
    same_function_name = 1 if functions_in_code1 == 0 else normalize(same_func, -max_functions, max_functions)
    same_method_name = 1 if methods_in_code1 == 0 else normalize(same_method, -methods_in_code1, methods_in_code1)

    # ---------------- Return similarity -----------------
    # Build lists of return expression AST nodes from each code snippet.
    returns1 = []
    returns2 = []

    for node in tree1.body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    returns1.append(stmt.value)

    for node in tree2.body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    returns2.append(stmt.value)

    # For return similarity, try to match each return expression from code1 with any (unused) one in code2.
    same_function_return = 1  # Start at 1.
    penalty = 0.35  # Penalty for each unmatched return.
    matched_indices = set()

    for r1 in returns1:
        match_found = False
        for i, r2 in enumerate(returns2):
            if i in matched_indices:
                continue
            # Compare AST dumps for structural equality.
            if ast.dump(r1) == ast.dump(r2):
                match_found = True
                matched_indices.add(i)
                break
        if not match_found:
            same_function_return -= penalty

    # ---------------- Dynamic Weighting -----------------
    # Only include components that exist in code1.
    weighted_sum = 0
    total_weight = 0

    if classes_in_code1 > 0:
        weighted_sum += class_sim * 1
        total_weight += 1

    if methods_in_code1 > 0:
        weighted_sum += meth_sim * 1
        total_weight += 1

    if expressions_in_code1 > 0:
        weighted_sum += exp_sim * 1
        total_weight += 1

    if functions_in_code1 > 0:
        weighted_sum += func_sim * 3     # structural weight
        weighted_sum += same_function_name * 2  # naming weight
        weighted_sum += same_function_return * 1  # return weight
        total_weight += 3 + 2 + 1

    if total_weight == 0:
        total_weight = 1
        weighted_sum = 1

    score = abs(weighted_sum / total_weight)
    similarity_percentage = "{:.2f}".format(score * 100)
    return jsonify({"result": similarity_percentage})


if __name__ == "__main__":
    app.run(debug=True)