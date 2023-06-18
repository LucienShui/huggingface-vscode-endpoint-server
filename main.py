import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request

from Llm import Llm
from api_models import ChatCompletionRequestPayload, ChatCompletionApiResponse
from api_models import CodingRequestPayload, CodingApiResponse
from generators import GeneratorBase, CodeGenerator, ChatGenerator
from request_handler import RequestHandler
from util import get_parser, logger


def read_version():
    return (Path(__file__).parent / "VERSION").read_text().strip()


app: FastAPI = FastAPI(
    title="TNG Internal Starcoder",
    version=read_version()
)

request_handler: RequestHandler = None
api_type: str = None


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(request_handler.process_request_queue())


@app.post("/api/generate/", response_model=CodingApiResponse)
async def api(request: Request, request_payload: CodingRequestPayload) -> CodingApiResponse:
    if api_type != 'code':
        return CodeGenerator.generate_default_api_response(message=f"api-type mismatch. Server is running in {api_type} mode", status=422)
    return await request_handler.handle_request(request, request_payload)


@app.post("/v1/chat/completions/", response_model=ChatCompletionApiResponse)
async def chat(request: Request, request_payload: ChatCompletionRequestPayload) -> ChatCompletionApiResponse:
    if api_type != 'chat':
        return ChatGenerator.generate_default_api_response(message=f"api-type mismatch. Server is running in {api_type} mode", status=422)
    return await request_handler.handle_request(request, request_payload)


def main():
    global request_handler, api_type
    args = get_parser().parse_args()
    api_type = args.api_type
    ssl_certificate, ssl_key = None, None
    if args.ssl_certificate and args.ssl_keyfile:
        ssl_certificate = args.ssl_certificate
        ssl_key = args.ssl_keyfile

    generator: GeneratorBase = None
    llm: Llm = Llm(model_name=args.pretrained, bitsize=args.bit_precission, do_not_load_llm=args.dry_run)
    if api_type == 'code':
        generator = CodeGenerator(llm=llm)
    elif args.api_type == 'chat':
        generator = ChatGenerator(llm=llm)
    else:
        logger.error(f"api_type {args.api_type} not supported. Use 'code' or 'chat'")
        exit()

    request_handler = RequestHandler(generator=generator, auth_prefix=args.auth_prefix)
    uvicorn.run(app, host=args.host, port=args.port, ssl_keyfile=ssl_key, ssl_certfile=ssl_certificate)


if __name__ == "__main__":
    main()
