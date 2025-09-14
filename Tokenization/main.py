import nltk
import spacy
from transformers import BertTokenizer, XLNetTokenizer
from datetime import datetime
from collections import Counter
import os


def ensure_nltk():
    # Ensure 'punkt'
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # Ensure 'punkt_tab' (newer NLTK versions may require it)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab')
        except Exception:
            # Silently ignore if resource not present in this NLTK release
            pass


def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


def show_frequencies(tokens, label):
    freq = Counter(tokens)
    print(f"Top frequencies for {label} (token -> count):")
    for token, count in freq.most_common(10):
        print(f"  {token!r}: {count}")
    print()


def main():
    text = os.environ.get(
        "TOKENIZATION_SAMPLE_TEXT",
        "IBM taught me tokenization. Tokenization helps models understand text."
    )

    print("Input Text:\n", text, "\n")

    ensure_nltk()
    nlp = ensure_spacy_model("en_core_web_sm")

    # NLTK Tokenization
    start_time = datetime.now()
    nltk_tokens = nltk.word_tokenize(text)
    nltk_time = datetime.now() - start_time

    # SpaCy Tokenization
    start_time = datetime.now()
    spacy_tokens = [token.text for token in nlp(text)]
    spacy_time = datetime.now() - start_time

    # BertTokenizer Tokenization
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    start_time = datetime.now()
    bert_tokens = bert_tokenizer.tokenize(text)
    bert_time = datetime.now() - start_time

    # XLNetTokenizer Tokenization
    xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    start_time = datetime.now()
    xlnet_tokens = xlnet_tokenizer.tokenize(text)
    xlnet_time = datetime.now() - start_time

    # Display tokens, time taken for each tokenizer, and token frequencies
    print(f"NLTK Tokens: {nltk_tokens}\nTime Taken: {nltk_time.total_seconds():.6f} seconds\n")
    show_frequencies(nltk_tokens, "NLTK")

    print(f"SpaCy Tokens: {spacy_tokens}\nTime Taken: {spacy_time.total_seconds():.6f} seconds\n")
    show_frequencies(spacy_tokens, "SpaCy")

    print(f"Bert Tokens: {bert_tokens}\nTime Taken: {bert_time.total_seconds():.6f} seconds\n")
    show_frequencies(bert_tokens, "Bert")

    print(f"XLNet Tokens: {xlnet_tokens}\nTime Taken: {xlnet_time.total_seconds():.6f} seconds\n")
    show_frequencies(xlnet_tokens, "XLNet")


if __name__ == "__main__":
    main()
