import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_reply(text: str) -> str:
    _load()
    inputs = _tokenizer.encode(text, return_tensors="pt")
    outputs = _model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def run_single(prompt: str):
    print(f"You: {prompt}")
    print("Chatbot:", generate_reply(prompt))


def run_batch(lines):
    for line in lines:
        line = line.strip()
        if not line:
            continue
        print(f"You: {line}")
        print("Chatbot:", generate_reply(line))


def run_interactive():
    while True:
        try:
            user = input("You: ")
        except EOFError:
            print("Chatbot: EOF. Goodbye!")
            break
        except KeyboardInterrupt:
            print("\nChatbot: Interrupted. Goodbye!")
            break
        if user.lower() in {"quit", "exit", "bye"}:
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", generate_reply(user))


def main():
    single = os.getenv("SINGLE_PROMPT")
    if single:
        run_single(single)
        return

    # Non‑TTY: consume any piped stdin; if none, show usage and exit
    if not sys.stdin.isatty():
        piped = [l.rstrip("\n") for l in sys.stdin]
        if piped:
            run_batch(piped)
        else:
            print("No TTY detected and no input provided.\n"
                  "Usage:\n"
                  "  docker run -it --rm image            # interactive\n"
                  "  echo 'Hi' | docker run -i --rm image  # piped batch\n"
                  "  docker run --rm -e SINGLE_PROMPT='Hi' image")
        return

    run_interactive()


if __name__ == "__main__":
    main()

# You: What is the Capitol of Texas?
# Chatbot: texas capitol
# You: What is the capital of Oklahoma?
# Chatbot: oklahoma city
# You: What is the capital of Colorado?
# Chatbot: st. louis
# You: