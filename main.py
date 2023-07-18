import uvicorn
from fastapi import FastAPI, Request, Header, Depends, HTTPException
from starlette import status
from fastapi.middleware.cors import CORSMiddleware
from generators import GeneratorBase, StarCoder, SantaCoder
import json

from pydantic import BaseModel
from util import logger, get_parser

app = FastAPI()
app.add_middleware(
    CORSMiddleware
)
generator: GeneratorBase = ...

# Placeholder for a database containing valid token values
known_tokens = set(["api_token_abc123"])


class UnauthorizedMessage(BaseModel):
    detail: str = "Bearer token missing or unknown"


async def get_token(
    authorization: str = Header(default="Bearer "),
) -> str:
    _, token = authorization.split(" ")
    # Simulate a database query to find a known token
    if token not in known_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UnauthorizedMessage().detail,
        )
    return token

@app.post("/api/generate/")
async def api(request: Request, token:str = Depends(get_token)):
    print(token)
    json_request: dict = await request.json()
    inputs: str = json_request['inputs']
    parameters: dict = json_request['parameters']
    print("\n\n",request.headers,"\n\n")
    logger.info(f'{request.client.host}:{request.client.port} inputs = {json.dumps(inputs)}')
    generated_text: str = generator.generate(inputs, parameters)
    logger.info(f'{request.client.host}:{request.client.port} generated_text = {json.dumps(generated_text)}')
    
    return {
        "generated_text": generated_text,
        "status": 200
    }

def main():
    global generator
    args = get_parser().parse_args()
    generator = SantaCoder(args.pretrained) # StarCoder(args.pretrained, device_map='auto')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()  