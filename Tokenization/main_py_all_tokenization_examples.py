def main():
    import nltk
    nltk.download("punkt")
    nltk.download('punkt_tab')
    import spacy  # noqa: F401
    from nltk.tokenize import word_tokenize  # noqa: F401
    from nltk.probability import FreqDist  # noqa: F401
    from nltk.util import ngrams  # noqa: F401
    from transformers import BertTokenizer  # noqa: F401
    from transformers import XLNetTokenizer  # noqa: F401
    from torchtext.data.utils import get_tokenizer  # noqa: F401
    from torchtext.vocab import build_vocab_from_iterator  # noqa: F401

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    warnings.filterwarnings('ignore')

    print("\nWord-Based Tokenization Examples\n")
    text1 = "This is a sample sentence for word tokenization.\n"
    print("This showcases word_tokenize from nltk library:")
    print(text1)
    print("Without punctuations:")
    tokens = word_tokenize(text1)
    print(tokens)

    text2 = "\nI couldn't help the dog. Can't you do it? Don't be afraid if you are."
    print(text2)
    print("With punctuations:")
    tokens_with_punctuations = word_tokenize(text2)
    print(tokens_with_punctuations)

    text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
    print("\nThis showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function")
    print(text)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Making a list of the tokens and priting the list
    token_list = [token.text for token in doc]
    print("Tokens:", token_list)

    # Showing token details
    for token in doc:
        print(token.text, token.pos_, token.dep_)

    text3 = "Unicorns are real. I saw a unicorn yesterday."
    token1 = word_tokenize(text3)
    print("Tokens:")
    print(token1)

    print("WordPiece Tokenization Example\n")
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized = tokenizer_bert.tokenize("IBM taught me tokenization.")
    print(tokenized)

    print("\nUnigram and SentencePiece Tokenization Example\n")
    tokenizer_xlnet = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    tokenized2 = tokenizer_xlnet.tokenize("IBM taught me tokenization.")
    print(tokenized2)

    # ---------------- Simple TorchText Vocabulary Demo (with OOV) -----------------
    print("\n=== TorchText Vocabulary & OOV Demonstration ===")
    dataset = [
        (1, "Introduction to NLP"),
        (2, "Basics of PyTorch"),
        (1, "NLP Techniques for Text Classification"),
        (3, "Named Entity Recognition with PyTorch"),
        (3, "Sentiment Analysis using PyTorch"),
        (3, "Machine Translation with PyTorch"),
        (1, " NLP Named Entity,Sentiment Analysis,Machine Translation "),
        (1, " Machine Translation with NLP "),
        (1, " Named Entity vs Sentiment Analysis  NLP "),
    ]

    # Create a basic English tokenizer
    tokenizer_basic = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for _, iterated_text in data_iter:
            yield tokenizer_basic(iterated_text)

    # Build vocabulary with <unk> token for OOV handling
    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print(f"Vocabulary size: {len(vocab)}")
    print("First 25 vocab entries:", vocab.get_itos()[:25])

    # Function provided in user's prompt adapted here
    def get_tokenized_sentence_and_indices(iterator):
        tokenized_sentence = next(iterator)  # next token list
        token_indices = [vocab[token] for token in tokenized_sentence]
        return tokenized_sentence, token_indices

    token_iterator = yield_tokens(dataset)
    tokenized_sentence, token_indices = get_tokenized_sentence_and_indices(token_iterator)

    print("\nSample tokenized sentence (first entry tokens):", tokenized_sentence)
    print("Corresponding indices:", token_indices)

    # Show another sentence from iterator for clarity
    try:
        second_sentence_tokens = next(token_iterator)
        print("Second sentence tokens:", second_sentence_tokens)
        print("Second sentence indices:", [vocab[t] for t in second_sentence_tokens])
    except StopIteration:
        pass

    # Demonstrate OOV handling with clearly unseen words
    oov_text = "Quantum unicorn pineapple appears"
    oov_tokens = tokenizer_basic(oov_text)
    oov_indices = [vocab[t] for t in oov_tokens]
    print("\nOOV Example Text:", oov_text)
    print("OOV Tokens:", oov_tokens)
    print("OOV Token Indices (unknown -> index of <unk>):", oov_indices)
    print(f"<unk> token index is: {vocab['<unk>']}")

    # Full dataset tokenization (all entries) for completeness
    all_token_lists = list(yield_tokens(dataset))
    print("\nAll dataset token lists:")
    for i, tl in enumerate(all_token_lists, start=1):
        print(f"Entry {i}:", tl)

    # Show mapping for a specific known token (e.g., 'nlp' if present)
    sample_token = 'nlp'
    print(f"\nIndex for token '{sample_token}':", vocab[sample_token])

    lines = ["IBM taught me tokenization",
             "Special tokenizers are ready and they will blow your mind",
             "just saying hi!"]

    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')

    tokens = []
    max_length = 0

    for line in lines:
        tokenized_line = tokenizer_en(line)
        tokenized_line = ['<bos>'] + tokenized_line + ['<eos>']
        tokens.append(tokenized_line)
        max_length = max(max_length, len(tokenized_line))

    for i in range(len(tokens)):
        tokens[i] = tokens[i] + ['<pad>'] * (max_length - len(tokens[i]))

    print("Lines after adding special tokens:\n", tokens)

    # Build vocabulary without unk_init
    vocab = build_vocab_from_iterator(tokens, specials=['<unk>'])
    vocab.set_default_index(vocab["<unk>"])

    # Vocabulary and Token Ids
    print("Vocabulary:", vocab.get_itos())
    print("\nToken IDs for 'tokenization':", vocab.get_stoi())

    new_line = "I learned about embeddings and attention mechanisms."

    # Tokenize the new line
    tokenized_new_line = tokenizer_en(new_line)
    tokenized_new_line = ['<bos>'] + tokenized_new_line + ['<eos>']

    # Pad the new line to match the maximum length of previous lines
    new_line_padded = tokenized_new_line + ['<pad>'] * (max_length - len(tokenized_new_line))

    # Convert tokens to IDs and handle unknown words
    new_line_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in new_line_padded]

    # Example usage
    print("Token IDs for new line:", new_line_ids)

if __name__ == "__main__":
    main()
