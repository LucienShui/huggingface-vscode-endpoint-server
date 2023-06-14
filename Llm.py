import torch
import time
from timeit import default_timer as timer
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
from util import logger

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords_ids = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords_ids:
            return True
        return False

class Llm:
    models = {'llama-small': {'model': 'decapoda-research/llama-7b-hf'},
            'llama-medium': {'model': 'decapoda-research/llama-13b-hf'},
            'llama-large': {'model': 'decapoda-research/llama-65b-hf'},
            'stable-vicuna': {'model': 'Llama-delta/stable-vicuna-13b'},
            'wizard-vicuna': {'model': 'TheBloke/Wizard-Vicuna-13B-Uncensored-HF'},
            'guanaco': {'model': 'JosephusCheung/Guanaco'},
            'guanaco-large': {'model': 'timdettmers/guanaco-65b'},
            'falcon-instruct-small': {'model': 'tiiuae/falcon-7b-instruct', 'trust_remote_code': True},
            'falcon-instruct-large': {'model': 'tiiuae/falcon-40b-instruct', 'trust_remote_code': True},
            'falcon-large': {'model': 'tiiuae/falcon-40b', 'trust_remote_code': True},
            'starcoder': {'model': 'bigcode/starcoder'},
            'testing': {'model': 'gpt2'},
            }

    bitsize_map = {8: {'load_in_8bit': True, 'torch_dtype': torch.float16},
                16: {'torch_dtype': torch.bfloat16},
                32: {'torch_dtype': torch.float}}

    def __init__(self, model_name, bitsize=16, device=None, do_not_load_llm=False):
        self.prev_time = timer()
        self.delta_t = 0
        self.timeit()
        self.device = "cuda" if device is None else device
        assert model_name in Llm.models, f"model {model_name} not found.\nchose one of: {[key for key in Llm.models.keys()]}"
        self.model_name = model_name
        self.model_config = self.get_model_config(model_name, bitsize)
        self.stopping_criteria_config = {}
        self.stop_ids = []
        self.load_tokenizer()
        # load model should be the last action, so that get_timing returns the load time if called after Llm()
        if do_not_load_llm:
            self.model = None
            return
        self.load_model(bitsize)

    def get_model_config(self, model_id: str, bitsize: int):
        config = Llm.models[model_id].copy()
        config.update(Llm.bitsize_map[bitsize])
        return config

    def get_model_parameters(self, bitsize):
        model_id = self.model_config['model']
        model_loader_class = AutoModelForCausalLM if "llama" not in model_id.lower() else LlamaForCausalLM
        model_loader = model_loader_class.from_pretrained

        params = {"device_map": self.get_device_map()}
        for param, value in self.model_config.items():
            if param == 'model':
                continue
            params[param] = value
        if bitsize == 8:
            params['load_in_8bit'] = True
        if bitsize == 4:
            params['quant_config'] = Llm.quant_config
        return model_loader, model_id, params

    def load_model(self, bitsize):
        model_loader, model_id, params = self.get_model_parameters(bitsize)
        logger.debug(f"loading model {model_id} using these parameters: {params}")
        self.timeit()
        self.model = model_loader(model_id, **params)
        self.timeit("load model")
        logger.debug(self.model.hf_device_map)
        self.print_model_layer_information()

    def load_tokenizer(self):
        model_id = self.model_config['model']
        tokenizer_loader = AutoTokenizer.from_pretrained if "llama" not in model_id.lower() else LlamaTokenizer.from_pretrained
        self.timeit()
        self.tokenizer = tokenizer_loader(model_id)
        self.timeit("load tokenizer")
        
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def add_stopwords(self, stop_word_list):
        stop_encoded = [self.tokenizer.encode(w) for w in stop_word_list]
        self.stop_ids = [encoded[0] for encoded in stop_encoded]
        stopping_criteria = KeywordsStoppingCriteria(self.stop_ids)
        self.stopping_criteria_config = {'stopping_criteria': StoppingCriteriaList([stopping_criteria])}

    def print_model_layer_information(self):
        size_dict = defaultdict(int)
        device_dict = defaultdict(set)
        layer_names = []
        for name, param in self.model.named_parameters():
            #print(name, param.size())
            layer_parts = name.split(".")
            layer_name = ".".join(layer_parts[0:3])
            if layer_name not in layer_names:
                layer_names.append(layer_name)
            size_dict[layer_name] += param.numel()
            device_dict[layer_name].add(param.device)
        for layer_name in layer_names:
            logger.debug(f"Layer: {layer_name}: {size_dict[layer_name]} on device {device_dict[layer_name]}")

    def strip_inputs_and_stopwords(self, outputs, input_ids):
        # remove the last token, if it was a stopping token
        end_idx = -1
        eol_id = self.tokenizer.encode("\n")[0]
        while outputs[0, end_idx] in self.stop_ids + [self.tokenizer.eos_token_id, eol_id]:
            end_idx -= 1
        end_idx = end_idx + 1 if end_idx < -1 else len(outputs[0])
        logger.debug(f"Stopping at {end_idx} token from the end")
        # remove input_ids from the outputs
        if outputs[0,0] != input_ids[0,0]:
            logger.warning(f"outputs do not start with input tokens, skipping stripping {outputs[0,0]} {input_ids[0,0]}")
            outputs = outputs[:, :end_idx]
        else:
            logger.debug(f"Stripping {len(input_ids[0])} tokens from the start")
            outputs = outputs[:, len(input_ids[0]):end_idx]
        return outputs

    def generate_from_ids(self, input_ids, generation_config: dict, remove_prompt_from_reply: bool=True) -> tuple:
        #if self.max_input_tokens is not None and input_ids.shape[1] > self.max_input_tokens:
        #    input_ids = input_ids[:, -self.max_input_tokens:]
        #    logger.info(f"reduced input token length to {input_ids.shape}, {self.max_input_tokens}")
        prompt_tokens = len(input_ids[0])
        self.timeit()
        if self.model is not None:
            outputs = self.model.generate(input_ids, **generation_config, **self.streamer_config, **self.stopping_criteria_config)
        else:
            outputs = input_ids
        self.timeit(f"inference {prompt_tokens}/{len(outputs[0]) - prompt_tokens}")
        if remove_prompt_from_reply:
            outputs = self.strip_inputs_and_stopwords(outputs, input_ids)
        completion_tokens = len(outputs[0])
        return outputs, prompt_tokens, completion_tokens

    def chat(self, prompt: str, generation_config: dict, remove_prompt_from_reply: bool=True, do_stream: bool=False) -> tuple:
        if self.model is None:
            return "Testing without LLM", 0, 0

        if do_stream and len(self.streamer_config) == 0:
            streamer = TextStreamer(self.tokenizer)
            self.streamer_config = {'streamer': streamer}

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs, prompt_tokens, completion_tokens = self.generate_from_ids(inputs['input_ids'], generation_config, remove_prompt_from_reply)
        answer = self.tokenizer.batch_decode(outputs)
        return answer[0].lstrip(), prompt_tokens, completion_tokens

    def timeit(self, label=None):
        cur_time = timer()
        self.delta_t = cur_time - self.prev_time
        if label is not None:
            print(label, self.delta_t)
        self.prev_time = cur_time

    def get_timing(self):
        return self.delta_t        

    def get_device_map(self):
        device_map = "auto"
        return device_map
    

