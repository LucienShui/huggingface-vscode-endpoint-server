# Hugging Face VSCode Endpoint Server

A server for [huggingface-vscdoe](https://github.com/huggingface/huggingface-vscode) custom endpoints using LLMs.

Now properly handles multiple client requests.
* still uses http
* not using batches for inference
  * for a single client, it's not required
  * multi-client use may benefit

## Usage

```shell
pip install -r requirements.txt
python main.py
```

Fill `http://localhost:8000/api/generate/` into `Hugging Face Code > Model ID or Endpoint` in VSCode.

In VS code: 
* "Hugging Face Code: Set API token" (type Ctrl + Shift + P)
* Set it according to the option: --auth_prefix, defaults to "<secret-key>"

## API

```shell
curl -X POST http://localhost:8000/api/generate/ -d '{"inputs": "def fib(n):", "parameters": {"max_new_tokens": "10"}}' -H "Authorization: Bearer <secret-key>"
# response = {"generated_text": "def fib(n):\n    if n == 0:\n        return"}
```
