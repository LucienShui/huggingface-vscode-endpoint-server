import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import GenerationConfig, pipeline, Pipeline
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    import accelerate
except (ImportError, ModuleNotFoundError):
    logger.warning('accelerate not installed, would not dispatch model to GPUs automatically.')

app = FastAPI()

app.add_middleware(
    CORSMiddleware
)

pretrained = "bigcode/starcoder"
generation_config: GenerationConfig = GenerationConfig.from_pretrained(pretrained)
pipe: Pipeline = pipeline("text-generation", model=pretrained, torch_dtype=torch.bfloat16, device_map='auto')


@app.post("/api/generate/")
async def chat(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    config = GenerationConfig.from_dict({
        **generation_config.to_dict(),
        **{"pad_token_id": pipe.tokenizer.eos_token_id},
        **parameters}
    )
    logger.info(f'{request.client.host}:{request.client.port} inputs = {inputs}')
    generated_text: str = pipe(inputs, generation_config=config)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {generated_text}')
    return {
        "generated_text": generated_text,
        "status": 200
    }


def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
