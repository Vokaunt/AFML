import os
import ast
from typing import Dict, Set


def _collect_functions(tree: ast.AST) -> Set[str]:
    funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs.add(node.name)
    return funcs


def parse_functions(directory: str) -> Dict[str, Set[str]]:
    """Parse functions and their calls inside a directory."""
    modules = {}
    for fname in os.listdir(directory):
        if fname.endswith('.py'):
            path = os.path.join(directory, fname)
            module = os.path.splitext(fname)[0]
            with open(path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            funcs = _collect_functions(tree)
            for fn in funcs:
                modules[f"{module}.{fn}"] = set()

    # second pass to collect calls
    for fname in os.listdir(directory):
        if fname.endswith('.py'):
            path = os.path.join(directory, fname)
            module = os.path.splitext(fname)[0]
            with open(path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    caller = f"{module}.{node.name}"
                    called = _find_called_functions(node.body, modules.keys())
                    modules[caller].update(called)
    return modules


def _find_called_functions(nodes, valid: Set[str]) -> Set[str]:
    calls = set()
    for node in ast.walk(ast.Module(body=nodes)):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if isinstance(node.func.value, ast.Name):
                    name = f"{node.func.value.id}.{attr}"
                else:
                    name = attr
            if name:
                for func in valid:
                    if func == name or func.endswith(f".{name}"):
                        calls.add(func)
    return calls
