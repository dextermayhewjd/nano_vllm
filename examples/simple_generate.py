
def main():
    l

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]
    
    lm = NanoLLM(model="gpt2")




    outs = llm.generate(
        ["Hello, my name is", "Artificial intelligence will"],
        max_new_tokens=20
    )

    for i, o in enumerate(outs):
        print(f"[Result {i}]")
        print(o)
        print()

if __name__ == "__main__":
    main()