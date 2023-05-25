import asyncio
import threading
from collections import deque
from fastapi import Request
from util import logger
from generators import GeneratorBase

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
                return auth_header[7:]
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
            await asyncio.sleep(.05)

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
                # strip off "<fim_prefix>" and "<fim_suffix>"
                start_idx = 0 if len(inputs) <= 24 else 12
                end_idx = len(inputs) if len(inputs) < 24 else len(inputs)-12
                logger.info(f" cache hit <{inputs[start_idx:start_idx+10]}...{inputs[end_idx-10:end_idx]}>")
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
            if generated_text is None:
                if self.generator is not None:
                    generated_text = await self.generator.generate(inputs, parameters)
                else:
                    generated_text = "testing without LLM"
                await self.response_cache.update(inputs, generated_text)
            logger.debug(f"done processing request from queue {request.client.port}")
            result: dict = {"generated_text": generated_text, "status": 200}
            request.state.result = result
            client_request.event.set()
    
    async def handle_request(self, request: Request) -> dict:
        self.cnt += 1
        local_cnt = self.cnt

        logger.debug(f"received request {local_cnt} from {request.client.host}:{request.client.port}")
        client_request: ClientRequest = await self.generate_client_request(request)

        if not client_request.id.startswith(self.auth_prefix):
            return {"generated_text": "no valid bearer token", "status": 401}

        generated_text = await self.response_cache.retrieve(client_request.inputs)
        if generated_text is not None:
            return {"generated_text": generated_text, "status": 200}

        exchanged_client_request = await self.queue.put_or_exchange(client_request)
        if exchanged_client_request is not None:
            logger.debug(f" expired request from port {exchanged_client_request.request.client.port}")
            result = {"generated_text": "", "status": 429}
            exchanged_client_request.request.state.result = result
            exchanged_client_request.event.set()
    
        logger.debug(f"waiting for {local_cnt} from port {request.client.port}")
        await client_request.event.wait()

        logger.debug(f"return request {local_cnt} from port {request.client.port}")
        return request.state.result

    async def generate_client_request(self, request: Request) -> ClientRequest:
        json_request: dict = await request.json()
        inputs: str = json_request['inputs']
        parameters: dict = json_request['parameters']
        return ClientRequest(request, inputs, parameters)
