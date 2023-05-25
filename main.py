import uvicorn
from fastapi import FastAPI, Request
import asyncio
from generators import GeneratorBase, HfAutoModelCoder
from request_handler import RequestHandler
from util import get_parser

app: FastAPI = FastAPI()
request_handler: RequestHandler = None

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(request_handler.process_request_queue())

@app.post("/api/generate/")
async def api(request: Request) -> dict:
    return await request_handler.handle_request(request)

def main():
    global request_handler
    args = get_parser().parse_args()
    generator: GeneratorBase = None
    generator = HfAutoModelCoder(pretrained=args.pretrained)
    request_handler = RequestHandler(generator=generator, auth_prefix=args.auth_prefix)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
