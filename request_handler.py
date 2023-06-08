import asyncio
import threading
from collections import deque
from fastapi import Request, HTTPException
import logging
import json
from util import logger
from generators import GeneratorBase, GeneratorException

class ClientRequest:
    def __init__(self, request: Request, inputs: str, parameters):
        self.id: str = self.get_client_id(request)
        self.request: Request = request
        self.inputs: str = inputs
        self.parameters = parameters
        self.event: asyncio.Event = asyncio.Event()

    def get_client_id(self, request):
        if 'authorization' in request._headers:
            auth_header = request._headers['authorization']
            logger.debug(f"auth_header {auth_header}")
            if auth_header.startswith("Bearer "):
                return auth_header[7:] + request.client.host
        return ""

class ClientRequestQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._client_items: dict = dict()
        self._lock: threading.Lock = threading.Lock()
        self._cache: dict = dict()
    
    async def put_or_exchange(self, item: ClientRequest) -> ClientRequest:
        client_id: str = item.id
        exchanged_item: ClientRequest = None
        with self._lock:
            if client_id in self._client_items:
                exchanged_item = self._client_items[client_id]
            else:
                self._queue.append(client_id)
            self._client_items[client_id] = item
        return exchanged_item

    async def get(self) -> ClientRequest:
        while True:
            with self._lock:
                if self._queue:
                    client_id = self._queue.popleft()
                    item = self._client_items.pop(client_id)
                    return item
            await asyncio.sleep(.01)

class ResponseCache:
    def __init__(self):
        self._cache: dict = dict()
        self._lock: threading.Lock = threading.Lock()

    async def update(self, inputs: str, generated_text: str):
        with self._lock:
            self._cache[inputs] = generated_text

    async def retrieve(self, inputs: str) -> str:
        with self._lock:
            if inputs in self._cache:
                if logger.isEnabledFor(logging.DEBUG):
                    # for debugging: strip off "<fim_prefix>" and "<fim_suffix>"
                    start_idx = 12 if inputs.startswith("<fim_prefix>") else 0
                    end_idx = len(inputs)-12 if inputs.endswith('<fim_middle>') else len(inputs)
                    start_str = inputs[start_idx:start_idx+10].replace("\n"," ")
                    end_str = inputs[end_idx-10:end_idx].replace("\n", " ")
                    logger.debug(f" cache hit <{start_str}>...<{end_str}>")
                generated_text: str = self._cache[inputs]
                return generated_text
        return None
    
class RequestHandler:
    def __init__(self, generator: GeneratorBase, auth_prefix: str):
        self.generator: GeneratorBase = generator
        self.queue: ClientRequestQueue = ClientRequestQueue()
        self.response_cache: ResponseCache = ResponseCache()
        self.auth_prefix = auth_prefix
        self.cnt = 0

    async def process_request_queue(self):
        while True:
            logger.debug("awaiting next request")
            client_request: ClientRequest = await self.queue.get()
            request: Request = client_request.request
            inputs: str = client_request.inputs
            parameters: str = client_request.parameters
            logger.debug(f"got request from queue {request.client.port}")
            generated_text = await self.response_cache.retrieve(inputs)
            try:
                if generated_text is None:
                    if self.generator is not None:
                        generated_text = await self.generator.generate(inputs, parameters)
                        await self.response_cache.update(inputs, generated_text)
                    else:
                        generated_text = "testing without LLM"
            except GeneratorException as e:
                # pass the error message as generated text, so that the user will see it within the IDE
                generated_text = str(e)
            logger.debug(f"done processing request from queue {request.client.port}")
            result: dict = {"generated_text": generated_text, "status": 200}
            request.state.result = result
            client_request.event.set()
    
    async def handle_request(self, request: Request) -> dict:
        self.cnt += 1
        local_cnt = self.cnt

        logger.info(f"received request {local_cnt} from {request.client.host}:{request.client.port}")
        try:
            client_request: ClientRequest = await self.generate_client_request(request)
        except json.decoder.JSONDecodeError as e:
            logger.debug(f"Can't parse request from {request.client.host}:{request.client.port} -> {e}")
            raise HTTPException(status_code=400, detail=f"Bad request: {e}")

        if not client_request.id.startswith(self.auth_prefix):
            logger.debug(f"request from {request.client.host}:{request.client.port} with invalid bearer token: '{client_request.id}'")
            # we don't throw an exception here, so that the user will see the error message within the IDE
            request.state.result = {"generated_text": "invalid bearer token", "status": 401}
            return request.state.result

        cached_text = await self.response_cache.retrieve(client_request.inputs)
        if cached_text is not None:
            request.state.result = {"generated_text": cached_text, "status": 200}
            return request.state.result

        exchanged_client_request = await self.queue.put_or_exchange(client_request)
        if exchanged_client_request is not None:
            logger.info(f" expired request from port {exchanged_client_request.request.client.port}")
            result = {"generated_text": "", "status": 429}
            exchanged_client_request.request.state.result = result
            exchanged_client_request.event.set()
    
        logger.debug(f"waiting for {local_cnt} from port {request.client.port}")
        await client_request.event.wait()

        logger.info(f"return request {local_cnt} from port {request.client.port}")
        return request.state.result
    
    

    async def generate_client_request(self, request: Request) -> ClientRequest:
        json_request: dict = await request.json()
        inputs: str = json_request['inputs']
        parameters: dict = json_request['parameters']
        return ClientRequest(request, inputs, parameters)
                               