# Hugging Face VSCode Endpoint Server

starcoder server for [huggingface-vscode](https://github.com/huggingface/huggingface-vscode) custom endpoint.

**Can't handle distributed inference very well yet.**

## Usage

```shell
pip install -r requirements.txt
python main.py
```

Fill `http://localhost:8000/api/generate/` into `Hugging Face Code > Model ID or Endpoint` in VSCode.

## API

```shell
curl -X POST http://localhost:8000/api/generate/ -d '{"inputs": "", "parameters": {"max_new_tokens": 64}}'
# response = {"generated_text": ""}
```
