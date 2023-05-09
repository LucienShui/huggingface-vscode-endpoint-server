import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import GenerationConfig

from util import logger, from_pretrained

app = FastAPI()

app.add_middleware(
    CORSMiddleware
)

pretrained = "bigcode/starcoder"
generation_config: GenerationConfig = GenerationConfig.from_pretrained(pretrained)
model, tokenizer = from_pretrained(pretrained)


def generate(inputs: str, parameters: dict) -> str:
    config = GenerationConfig.from_dict({
        **generation_config.to_dict(),
        **{"pad_token_id": tokenizer.eos_token_id},
        **parameters}
    )

    encoded_ids = tokenizer.encode(inputs, return_tensors="pt").to('cuda:0')
    output_ids = model.generate(encoded_ids, generation_config=config)
    response = tokenizer.decode(output_ids[0])
    return response


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    logger.info(f'{request.client.host}:{request.client.port} inputs = {inputs}')
    generated_text: str = generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {generated_text}')
    return {
        "generated_text": generated_text,
        "status": 200
    }


def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
