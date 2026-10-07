#!/usr/bin/env python3
"""Child-process evaluator for fixed local coding fixtures.

Runs inside the strict sandbox-exec profile written by benchmark.py (no network,
file reads limited to the Python runtime, system libraries and the task
workspace, writes limited to the workspace). That profile is the boundary; the
AST screen below is defense in depth and stays permissive enough that ordinary
correct code (helpers, annotations, defaults, small stdlib imports) is graded
on its behavior rather than its style.
"""

from __future__ import annotations

import ast
import builtins
import json
import resource
import sys
from pathlib import Path

ALLOWED_IMPORTS = {
    "collections", "dataclasses", "functools", "heapq", "itertools", "math",
    "re", "string", "typing", "bisect", "statistics", "enum", "operator",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "iter", "len", "list", "map", "max",
    "min", "next", "ord", "range", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "hash", "repr", "type", "object", "property",
    "staticmethod", "classmethod", "super", "issubclass", "callable", "format",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AssertionError", "StopIteration", "ZeroDivisionError", "NotImplementedError",
    "RuntimeError", "ArithmeticError", "LookupError", "OverflowError",
    "True", "False", "None", "NotImplemented", "__build_class__",
}


def emit(result_path: Path, passed: bool, detail: str) -> int:
    result_path.write_text(json.dumps({"passed": passed, "detail": detail}) + "\n")
    return 0 if passed else 1


def fail(result_path: Path, detail: str) -> int:
    return emit(result_path, False, detail)


def apply_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def validate(code: str, expected_name: str) -> tuple[bool, str, ast.Module | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"syntax error: {error}", None
    top_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if expected_name not in top_functions:
        return False, f"no top-level function named {expected_name}", None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.AsyncWith)):
            return False, f"forbidden construct: {type(node).__name__}", None
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"import not allowed: {alias.name}", None
        if isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False, f"import not allowed: {node.module}", None
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"forbidden dunder attribute: {node.attr}", None
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or (node.id.startswith("__") and node.id != "__name__")):
            return False, f"forbidden name: {node.id}", None
    return True, "validated", tree


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002 - mirrors __import__
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def main() -> int:
    if len(sys.argv) != 5:
        return 2
    candidate_path, tests_path, expected_name, result_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    code = candidate_path.read_text()
    tests = tests_path.read_text()
    valid, detail, tree = validate(code, expected_name)
    if not valid or tree is None:
        return fail(result_path, detail)
    # Pre-import the allowed modules before limits apply, then lock imports down.
    for module in ALLOWED_IMPORTS:
        __import__(module)
    apply_limits()
    allowed = {name: getattr(builtins, name) for name in SAFE_BUILTINS if hasattr(builtins, name)}
    allowed["__import__"] = guarded_import
    allowed["print"] = lambda *args, **kwargs: None
    namespace = {"__builtins__": allowed, "__name__": "candidate"}
    try:
        exec(compile(tree, str(candidate_path), "exec"), namespace)
        exec(compile(tests, str(tests_path), "exec"), namespace)
    except BaseException as error:
        return fail(result_path, f"fixed tests failed: {type(error).__name__}: {error}")
    return emit(result_path, True, "fixed tests passed in sandboxed evaluator")


if __name__ == "__main__":
    raise SystemExit(main())
