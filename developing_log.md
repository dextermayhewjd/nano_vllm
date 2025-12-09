开发日志

2025-12-9

- 今日处理wsl和shadowsocket

  1. 记得开allow other device to connect

  2. 但是同时也要记得把防火墙打开（在不下载的时候）

  3. 代理部分要考虑调回

  4. 其次需要条python interpreter

      4.1需要vscode支持 wsl版本的
  
      4.2需要确保能够pip能够使用 要装一个额外包才能用pysock好像
  
      4.3反复验证是否能够

  - # tokenizer 
  使用的是huggingface的autokenizer 
  ```python
    from transformers import AutoTokenizer

  # Download vocabulary from huggingface.co and cache.
  tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

  # Download vocabulary from huggingface.co (user-uploaded) and cache.
  tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

  # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
  # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")

  # Download vocabulary from huggingface.co and define model-specific arguments
  tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)

  # Explicitly use the tokenizers backend
  tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="tokenizers")

  # Explicitly use the sentencepiece backend
  tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", backend="sentencepiece")
  ```

  然后对于tokenizer的encode和decode 在这里  
  huggingface的encode和decode怎么用可以看[[https://huggingface.co/docs/transformers/v5.0.0rc0/zh/main_classes/tokenizer#transformers.TokenizersBackend.encode]]