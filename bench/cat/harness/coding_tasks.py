"""Fixed coding fixtures for the cat benchmark coding suite.

Every task is graded by executing the model's output against fixed unit tests
(Python, Swift) or by comparing a parsed JSON object (tool calls). Prompts are
plain user turns with no system prompt, matching how the host app sends a chat.
"""

from __future__ import annotations

from typing import Any

PYTHON = "python"
SWIFT = "swift"
JSON = "json"

ORIGINAL_TASKS = ("function_completion", "bug_fix", "simple_refactor", "json_tool_call")

CODING_TASKS: dict[str, dict[str, Any]] = {
    # The four 2026-10-02 fixtures, prompts unchanged.
    "function_completion": {
        "kind": PYTHON,
        "entry": "clamp",
        "prompt": "Output only a Python function named clamp(value, lower, upper). It must raise ValueError when lower > upper and otherwise constrain value to the inclusive interval.",
        "tests": "assert clamp(5, 0, 10) == 5\nassert clamp(-2, 0, 10) == 0\nassert clamp(12, 0, 10) == 10\ntry:\n clamp(1, 2, 0)\nexcept ValueError:\n pass\nelse:\n raise AssertionError('missing ValueError')",
    },
    "bug_fix": {
        "kind": PYTHON,
        "entry": "unique_counts",
        "prompt": "The function below wrongly drops duplicates. Output only a corrected Python function with the same name and stable ordering.\n\ndef unique_counts(items):\n    return {item: 1 for item in set(items)}",
        "tests": "assert unique_counts(['b', 'a', 'b', 'c', 'a']) == {'b': 2, 'a': 2, 'c': 1}\nassert list(unique_counts(['b', 'a', 'b', 'c', 'a'])) == ['b', 'a', 'c']\nassert unique_counts([]) == {}",
    },
    "simple_refactor": {
        "kind": PYTHON,
        "entry": "normalize_names",
        "prompt": "Refactor this Python function for clarity without changing behavior. Output only the complete function.\n\ndef normalize_names(items):\n    x = []\n    for i in items:\n        if i != None:\n            y = i.strip().lower()\n            if y != '':\n                x.append(y)\n    return x",
        "tests": "assert normalize_names([' Alice ', None, '', 'BOB']) == ['alice', 'bob']\nassert normalize_names(['  ', 'x']) == ['x']\nassert normalize_names([]) == []",
    },
    "json_tool_call": {
        "kind": JSON,
        "prompt": "Return only one JSON object representing a tool call. Exact schema: {\"tool\":\"lookup_weather\",\"arguments\":{\"city\":string,\"units\":\"celsius\"}}. The requested city is London.",
        "expected": {"tool": "lookup_weather", "arguments": {"city": "London", "units": "celsius"}},
    },
    # 2026-10-07 additions: small real-world functions, two bug fixes, two tool calls, two Swift.
    "parse_duration": {
        "kind": PYTHON,
        "entry": "parse_duration",
        "prompt": "Write a Python function parse_duration(text) that converts a duration string like \"1h30m\", \"45s\", \"2h\", \"1h2m3s\" or \"90m\" into a total number of seconds (int). Units are h, m and s, each used at most once, in that order. Surrounding whitespace is allowed. Raise ValueError for an empty string or anything else that does not match. Output only the code.",
        "tests": "assert parse_duration('1h30m') == 5400\nassert parse_duration('45s') == 45\nassert parse_duration('2h') == 7200\nassert parse_duration('1h2m3s') == 3723\nassert parse_duration(' 90m ') == 5400\nfor bad in ['', '10', 'h', '5x', '1m1h', '1h1h']:\n try:\n  parse_duration(bad)\n except ValueError:\n  pass\n else:\n  raise AssertionError('accepted ' + repr(bad))",
    },
    "merge_intervals": {
        "kind": PYTHON,
        "entry": "merge_intervals",
        "prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs (integers, start <= end, any order) and returns a new list of merged, non-overlapping [start, end] pairs sorted by start. Intervals that touch (end == next start) are merged. Do not mutate the input. Output only the code.",
        "tests": "src = [[8, 10], [1, 3], [2, 6], [15, 18]]\nassert merge_intervals(src) == [[1, 6], [8, 10], [15, 18]]\nassert src == [[8, 10], [1, 3], [2, 6], [15, 18]]\nassert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]",
    },
    "slugify": {
        "kind": PYTHON,
        "entry": "slugify",
        "prompt": "Write a Python function slugify(title) for blog URLs: lowercase the text, replace every run of characters that are not ASCII letters or digits with a single hyphen, and strip hyphens from both ends. Output only the code.",
        "tests": "assert slugify('Hello, World!') == 'hello-world'\nassert slugify('  Local AI -- Cat 2  ') == 'local-ai-cat-2'\nassert slugify('café au lait') == 'caf-au-lait'\nassert slugify('---') == ''\nassert slugify('already-a-slug') == 'already-a-slug'",
    },
    "roman_to_int": {
        "kind": PYTHON,
        "entry": "roman_to_int",
        "prompt": "Write a Python function roman_to_int(s) that converts a valid uppercase Roman numeral (1 to 3999) to an int. Output only the code.",
        "tests": "assert roman_to_int('III') == 3\nassert roman_to_int('IV') == 4\nassert roman_to_int('IX') == 9\nassert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994\nassert roman_to_int('MMMCMXCIX') == 3999",
    },
    "top_words": {
        "kind": PYTHON,
        "entry": "top_words",
        "prompt": "Write a Python function top_words(text, k) that returns the k most frequent words as a list of (word, count) tuples. Words are maximal runs of ASCII letters, compared case-insensitively and returned lowercase. Order by count descending, then alphabetically for ties. Output only the code.",
        "tests": "assert top_words('the cat and the hat and THE bat', 2) == [('the', 3), ('and', 2)]\nassert top_words('b a c b a', 3) == [('a', 2), ('b', 2), ('c', 1)]\nassert top_words(\"it's 2 cats, it's\", 2) == [('it', 2), ('s', 2)]\nassert top_words('', 3) == []",
    },
    "csv_line": {
        "kind": PYTHON,
        "entry": "parse_csv_line",
        "prompt": "Write a Python function parse_csv_line(line) that splits one CSV line into a list of string fields without using the csv module. Fields are separated by commas. A field may be wrapped in double quotes, in which case it can contain commas, and a doubled quote (\"\") inside it means one literal quote. Output only the code.",
        "tests": "assert parse_csv_line('a,b,c') == ['a', 'b', 'c']\nassert parse_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']\nassert parse_csv_line('\"say \"\"hi\"\"\",x') == ['say \"hi\"', 'x']\nassert parse_csv_line('a,,b') == ['a', '', 'b']\nassert parse_csv_line('') == ['']\nassert parse_csv_line('x,') == ['x', '']",
    },
    "bug_fix_binary_search": {
        "kind": PYTHON,
        "entry": "find_index",
        "prompt": "This function should return the index of target in the sorted list items, or -1 if it is absent, but it hangs or misses elements on some inputs. Fix it. Output only the corrected function.\n\ndef find_index(items, target):\n    lo, hi = 0, len(items)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return -1",
        "tests": "xs = [1, 3, 5, 7, 9, 11]\nfor i, x in enumerate(xs):\n assert find_index(xs, x) == i, x\nfor x in [0, 2, 4, 12]:\n assert find_index(xs, x) == -1, x\nassert find_index([], 3) == -1\nassert find_index([4], 4) == 0",
    },
    "bug_fix_chunks": {
        "kind": PYTHON,
        "entry": "chunks",
        "prompt": "chunks([1, 2, 3, 4, 5], 2) should return [[1, 2], [3, 4], [5]] but the function below loses the last partial chunk and accepts a size of 0. Fix it so the final partial chunk is kept and size < 1 raises ValueError. Output only the corrected function.\n\ndef chunks(items, size):\n    out = []\n    for i in range(0, len(items) - size + 1, size):\n        out.append(items[i:i + size])\n    return out",
        "tests": "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\nassert chunks([], 3) == []\nassert chunks([1], 5) == [[1]]\nfor bad in [0, -1]:\n try:\n  chunks([1], bad)\n except ValueError:\n  pass\n else:\n  raise AssertionError('accepted size ' + repr(bad))",
    },
    "tool_call_reminder": {
        "kind": JSON,
        "prompt": "You can call these tools:\n- create_reminder(title: string, due: {date: \"YYYY-MM-DD\", time: \"HH:MM\" 24-hour})\n- send_message(to: string, body: string)\n\nToday is Wednesday 2026-10-07. User: \"remind me to call mom tomorrow at 9am\".\n\nRespond with only a JSON object of the form {\"tool\": <name>, \"arguments\": {...}} for the single best tool call.",
        "expected": {"tool": "create_reminder", "arguments": {"title": "<contains:mom>", "due": {"date": "2026-10-08", "time": "09:00"}}},
    },
    "tool_call_typed_args": {
        "kind": JSON,
        "prompt": "Tool: search_files(query: string, limit: integer, include_archived: boolean, extensions: array of strings without dots).\n\nUser: \"find my invoice files, pdf or csv only, at most 5 results, skip anything archived\".\n\nRespond with only a JSON object {\"tool\": \"search_files\", \"arguments\": {...}} using correct JSON types.",
        "expected": {"tool": "search_files", "arguments": {"query": "<contains:invoice>", "limit": 5, "include_archived": False, "extensions": "<set:pdf,csv>"}},
    },
    "swift_format_duration": {
        "kind": SWIFT,
        "prompt": "Write a Swift function `func formatDuration(_ seconds: Int) -> String` that formats a non-negative number of seconds like a media player: \"m:ss\" under an hour (e.g. 65 -> \"1:05\", 5 -> \"0:05\") and \"h:mm:ss\" from one hour up (e.g. 3661 -> \"1:01:01\"). Output only the Swift code, no top-level statements.",
        "tests": "precondition(formatDuration(0) == \"0:00\")\nprecondition(formatDuration(5) == \"0:05\")\nprecondition(formatDuration(65) == \"1:05\")\nprecondition(formatDuration(599) == \"9:59\")\nprecondition(formatDuration(3599) == \"59:59\")\nprecondition(formatDuration(3600) == \"1:00:00\")\nprecondition(formatDuration(3661) == \"1:01:01\")\nprecondition(formatDuration(36000) == \"10:00:00\")\nprint(\"SWIFT_TESTS_PASSED\")",
    },
    "swift_median": {
        "kind": SWIFT,
        "prompt": "Write a Swift function `func median(_ values: [Double]) -> Double?` that returns nil for an empty array, the middle value for an odd count, and the mean of the two middle values for an even count. The input is not sorted and must not be mutated. Output only the Swift code, no top-level statements.",
        "tests": "precondition(median([]) == nil)\nprecondition(median([3]) == 3)\nprecondition(median([3, 1, 2]) == 2)\nprecondition(median([4, 1, 3, 2]) == 2.5)\nprecondition(median([-1, -5, 10, 0]) == -0.5)\nlet xs: [Double] = [5, 1, 4]\nprecondition(median(xs) == 4 && xs == [5, 1, 4])\nprint(\"SWIFT_TESTS_PASSED\")",
    },
}


def matches_expected(actual: Any, expected: Any) -> bool:
    """Exact JSON comparison with two loose markers for free-text fields.

    "<contains:x>" accepts any string containing x case-insensitively, and
    "<set:a,b>" accepts a list holding exactly those strings in any order.
    """
    if isinstance(expected, str) and expected.startswith("<contains:"):
        needle = expected[len("<contains:"):-1]
        return isinstance(actual, str) and needle.lower() in actual.lower()
    if isinstance(expected, str) and expected.startswith("<set:"):
        wanted = sorted(expected[len("<set:"):-1].split(","))
        return isinstance(actual, list) and all(isinstance(item, str) for item in actual) and sorted(actual) == wanted
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            matches_expected(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    return type(actual) is type(expected) and actual == expected
