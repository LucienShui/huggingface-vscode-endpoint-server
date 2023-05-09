import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import GenerationConfig, pipeline, Pipeline, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware
)


def get_device_map(max_memory: dict = None) -> dict:
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(pretrained))
    device_map = infer_auto_device_map(model, max_memory=max_memory)
    logger.info(f'device_map = {device_map}')
    return device_map


pretrained = "bigcode/starcoder"
generation_config: GenerationConfig = GenerationConfig.from_pretrained(pretrained)
pipe: Pipeline = pipeline("text-generation", model=pretrained, torch_dtype=torch.bfloat16, device_map=get_device_map())


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
