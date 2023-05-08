import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, Pipeline
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

pretrained = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
pipe: Pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

app = FastAPI()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware
)


@app.post("/api/generate/")
async def chat(request: Request):
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    config: GenerationConfig = GenerationConfig(**parameters)
    logger.info(f'{request.client.host}:{request.client.port} inputs = {inputs}')
    generated_text: str = pipe(inputs, generation_config=config)
    return {
        "generated_text": generated_text,
        "status": 200
    }


def main():
    uvicorn.run(app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
