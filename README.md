# Hugging Face VSCode Endpoint Server

A server for [huggingface-vscdoe](https://github.com/huggingface/huggingface-vscode) custom endpoints using LLMs.

This fork properly handles multiple client requests, adds Bearer-token authentication and supports https.

Currently, we are not using batches for inference
  * for a single client, it's not required
  * multi-client use may benefit

## Usage

```shell
pip install -r requirements.txt
python main.py
```

Use `http://localhost:8000/api/generate/` as `Hugging Face Code > Model ID or Endpoint` in VSCode.

In VS code: 
* "Hugging Face Code: Set API token" (type Ctrl + Shift + P)
* Set it according to the option: --auth-prefix, which defaults to "&lt;secret-key&gt;"

## API

```shell
curl -X POST http://localhost:8000/api/generate/ -d '{"inputs": "def fib(n):", "parameters": {"max_new_tokens": "10"}}' -H "Authorization: Bearer <secret-key>"
# response = {"generated_text": "def fib(n):\n    if n == 0:\n        return"}
```

## Completion triggers
The extension triggers, whenever one of the keys listed below gets typed. 

If the IDE does not show a suggestion after you typed key, you can retrigger it by typing DEL + key. As the server caches previous completions, this is more efficient than continuing to type.

```typescript
export const COMPLETION_TRIGGERS = [
  " ",
  ".",
  "(",
  ")",
  "{",
  "}",
  "[",
  "]",
  ",",
  ":",
  "'",
  '"',
  "=",
  "<",
  ">",
  "/",
  "\\",
  "+",
  "-",
  "|",
  "&",
  "*",
  "%",
  "=",
  "$",
  "#",
  "@",
  "!",
];
```
The source for the list of keys is [consts.ts](https://github.com/huggingface/huggingface-vscode/blob/master/src/globals/consts.ts).
