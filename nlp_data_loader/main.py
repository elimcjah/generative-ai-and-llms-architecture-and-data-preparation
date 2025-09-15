"""Minimal entrypoint for nlp_data_loader.
Verifies core library imports and prints their versions.
"""

import importlib

LIBS = [
    ("nltk", None),
    ("transformers", "__version__"),
    ("sentencepiece", "__version__"),
    ("spacy", "__version__"),
    ("numpy", "__version__"),
    ("torch", "__version__"),
    ("torchtext", "__version__"),
    # torchdata removed from requirements; keep optional check label only
]

OPTIONAL_LIBS = [
    ("torchdata", "__version__"),
]

def _version(mod, attr):
    if attr and hasattr(mod, attr):
        return getattr(mod, attr)
    return getattr(mod, "__version__", "unknown")

def main():
    results = []
    for name, attr in LIBS:
        try:
            m = importlib.import_module(name)
            results.append(f"{name}={_version(m, attr)}")
        except Exception as e:
            results.append(f"{name}=ERROR({e.__class__.__name__}: {e})")

    # Optional libs (do not count errors as failures)
    for name, attr in OPTIONAL_LIBS:
        try:
            m = importlib.import_module(name)
            results.append(f"(optional) {name}={_version(m, attr)}")
        except Exception:
            results.append(f"(optional) {name}=not installed")

    print("Loaded libs:")
    for line in results:
        print(" -", line)

if __name__ == "__main__":
    main()
