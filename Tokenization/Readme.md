docker build -t tokenization-app .
docker run --rm -it tokenization-app

Special Tokens:
Token: "<unk>", Index: 0: <unk> stands for "unknown" and represents words that were not seen during vocabulary building, usually during inference on new text.
Token: "<pad>", Index: 1: <pad> is a "padding" token used to make sequences of words the same length when batching them together.
Token: "<bos>", Index: 2: <bos> is an acronym for "beginning of sequence" and is used to denote the start of a text sequence.
Token: "<eos>", Index: 3: <eos> is an acronym for "end of sequence" and is used to denote the end of a text sequence.
Word Tokens: The rest of the tokens are words or punctuation extracted from the provided sentences, each assigned a unique index:
Token: "IBM", Index: 5
Token: "taught", Index: 16
Token: "me", Index: 12 ... and so on.
Vocabulary: It denotes the total number of tokens in the sentences upon which vocabulary is built.

Token IDs for 'tokenization': It represents the token IDs assigned in the vocab where a number represents its presence in the sentence.