from collections import defaultdict
from timeit import default_timer as timer

import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

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

    generation_config_overrides = {'falcon': {'ignore': ['stop']}}

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
 
        self.max_position_embeddings = None
        if hasattr(self.model.config,'max_position_embeddings'):
            self.max_position_embeddings = self.model.config.max_position_embeddings
            
        logger.debug(self.model.hf_device_map)
        self.print_model_layer_information()

    def load_tokenizer(self):
        model_id = self.model_config['model']
        tokenizer_loader = AutoTokenizer.from_pretrained if "llama" not in model_id.lower() else LlamaTokenizer.from_pretrained
        self.timeit()
        self.tokenizer = tokenizer_loader(model_id)
        self.timeit("load tokenizer")
        
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.device)

    def add_stopwords(self, stop_word_list):
        self.stopping_criteria_config = {'stopping_criteria': self.get_stopping_criteria_list(stop_word_list)}

    def get_stopping_criteria_list(self, stop_word_list) -> StoppingCriteriaList:
        stop_encoded = [self.tokenizer.encode(w) for w in stop_word_list]
        self.stop_ids = [encoded[0] for encoded in stop_encoded]
        stopping_criteria = KeywordsStoppingCriteria(self.stop_ids)
        return StoppingCriteriaList([stopping_criteria])

    def print_model_layer_information(self):
        size_dict = defaultdict(int)
        device_dict = defaultdict(set)
        layer_names = []
        for name, param in self.model.named_parameters():
            layer_parts = name.split(".")
            layer_name = ".".join(layer_parts[0:3])
            if layer_name not in layer_names:
                layer_names.append(layer_name)
            size_dict[layer_name] += param.numel()
            device_dict[layer_name].add(param.device)
        for layer_name in layer_names:
            logger.debug(f"Layer: {layer_name}: {size_dict[layer_name]/1024**2:.3f} MB on device {device_dict[layer_name]}")
        devices = {device for device_set in device_dict.values() for device in device_set}
        for device in devices:
            logger.debug(f"Used GPU mem: {sum([param.numel() for _, param in self.model.named_parameters() if device == param.device])/1024**3:.3f} GB on {device}")
        if len(devices) > 1:
            logger.debug(f"Used GPU mem: {sum(size_dict[l] for l in size_dict.keys())/1024**3:.3f} GB in total")

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

    def update_generation_config(self, generation_config):
        ignore_list = []
        for model_prefix, overrides in Llm.generation_config_overrides.items():
            if self.model_name.startswith(model_prefix):
                ignore_list += overrides['ignore']
        return {k: v for k,v in generation_config.items() if k not in ignore_list}

    def generate_from_ids(self, inputs, generation_config: dict, stopping_criteria_list: StoppingCriteriaList=None, remove_prompt_from_reply: bool=True) -> tuple:
        input_ids = inputs['input_ids']
        prompt_tokens = len(input_ids[0])
        generation_config = self.update_generation_config(generation_config)
        self.timeit()
        if self.model is not None:
            if self.max_position_embeddings is not None and prompt_tokens > self.max_position_embeddings:
                logger.debug(f"ignoring request: input sequence too long {prompt_tokens} > {self.max_position_embeddings}")
                return self.tokenize(f"input sequence too long {prompt_tokens} > {self.max_position_embeddings}")['input_ids'], prompt_tokens, 0
            if stopping_criteria_list is not None:
                stopping_criteria_config = {'stopping_criteria': stopping_criteria_list}
            elif self.stopping_criteria_config is not None:
                stopping_criteria_config = self.stopping_criteria_config
            else:
                stopping_criteria_config = {}
            if self.stopping_criteria_config is not None:
                stopping_criteria_config = self.stopping_criteria_config
            outputs = self.model.generate(**inputs, **generation_config, **stopping_criteria_config, pad_token_id=self.tokenizer.eos_token_id)
        else:
            outputs = input_ids
        self.timeit(f"inference {prompt_tokens}/{len(outputs[0]) - prompt_tokens}")
        if remove_prompt_from_reply:
            outputs = self.strip_inputs_and_stopwords(outputs, input_ids)
        completion_tokens = len(outputs[0])
        return outputs, prompt_tokens, completion_tokens

    def generate(self, prompt: str, generation_config: dict, stopping_criteria_list: StoppingCriteriaList=None, remove_prompt_from_reply: bool=True) -> tuple:
        if self.model is None:
            return "Testing without LLM", 0, 0

        inputs = self.tokenize(prompt)
        outputs, prompt_tokens, completion_tokens = self.generate_from_ids(inputs, generation_config, stopping_criteria_list, remove_prompt_from_reply)
        answer = self.tokenizer.batch_decode(outputs)
        return answer[0].lstrip(), prompt_tokens, completion_tokens

    def timeit(self, label=None):
        cur_time = timer()
        self.delta_t = cur_time - self.prev_time
        if label is not None:
            logger.debug(f"{label}: {self.delta_t}")
        self.prev_time = cur_time

    def get_timing(self):
        return self.delta_t        

    def get_device_map(self):
        device_map = "auto"
        return device_map
    

