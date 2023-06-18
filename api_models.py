from typing import Optional, List

from pydantic import BaseModel


class GeneratorException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class GeneratorBase:
    def generate(self, request_payload: BaseModel) -> BaseModel:
        raise NotImplementedError

    def generate_default_api_response(message: str, status: int) -> BaseModel:
        raise NotImplementedError


class CodingParameters(BaseModel):
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0
    do_sample: Optional[bool] = False
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None

    def key(self):
        return (self.max_new_tokens, self.temperature, self.do_sample, self.top_p, tuple(self.stop))


class CodingRequestPayload(BaseModel):
    inputs: str
    parameters: Optional[CodingParameters]

    def key(self):
        return (self.inputs, self.parameters.key if self.parameters else "")


class CodingApiResponse(BaseModel):
    generated_text: str
    status: int


class CompletionRequestPayload(BaseModel):
    model: str
    prompt: str = "<|endoftext|>"
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[float] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequence_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

    def key(self):
        return (self.model, self.prompt, self.max_tokens, self.temperature, self.user)


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str]


class ChatCompletionRequestPayload(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[float] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequence_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = "anonymous"

    def key(self):
        return (self.model, "\n".join([f"{role}{name}: {content}" for role, content, name in self.messages]), self.user)


class CompletionApiChoice(BaseModel):
    text: str
    index: int
    logprobs: List[float]
    finish_reason: str


class ApiUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionApiResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionApiChoice]
    usage: ApiUsage


class ChatCompletionApiChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionApiResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionApiChoice]
    usage: ApiUsage
