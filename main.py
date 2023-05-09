import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import GenerationConfig, Pipeline, pipeline
import torch
import json

from util import logger, get_parser

app = FastAPI()
app.add_middleware(
    CORSMiddleware
)

generation_config: GenerationConfig = ...
pipe: Pipeline = ...


def generate(inputs: str, parameters: dict) -> str:
    config = GenerationConfig.from_dict({
        **generation_config.to_dict(),
        **{"pad_token_id": pipe.tokenizer.eos_token_id},
        **parameters}
    )
    json_response: dict = pipe(inputs, generation_config=config)[0]
    generated_text: str = json_response['generated_text']
    return generated_text


@app.post("/api/generate/")
async def api(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')
    generated_text: str = generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    return {
        "generated_text": generated_text,
        "status": 200
    }


def main():
    global generation_config, pipe
    args = get_parser().parse_args()
    generation_config = GenerationConfig.from_pretrained(args.pretrained)
    pipe = pipeline("text-generation", model=args.pretrained, torch_dtype=torch.float16, device_map='auto')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
