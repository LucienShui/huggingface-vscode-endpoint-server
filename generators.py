from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import Pipeline, pipeline
import torch
from util import logger
import time


class GeneratorBase:
    def generate(self, query: str, parameters: dict) -> str:
        raise NotImplementedError

    def __call__(self, query: str, parameters: dict = None) -> str:
        return self.generate(query, parameters)

    def timeit(self, prompt=None):
        if prompt is None:
            self.t0 = time.perf_counter()
        else:
            t1 = time.perf_counter()
            logger.info(f"  time for '{prompt}': {t1-self.t0:.4f}")
            self.t0 = t1

    def sanitize_parameters(self, parameters: dict, parameter_key_types: dict) -> dict:
        p_dict = {}
        for key in parameters.keys():
            if key in parameter_key_types:
                value = parameters[key]
                conversion_func = parameter_key_types[key]
                try:
                    p_dict[key] = conversion_func(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Cannot convert parameter {key}:{value} to type {conversion_func}: {e}")
            else:
                logger.info(f"generator.py: ignoring client parameter {key}:{value} No datatype has been configured yet.")
        return p_dict


class HfAutoModelCoder(GeneratorBase):
    parameter_key_types = {'max_new_tokens': int, 
                           'temperature': float,
                           'do_sample': bool,
                           'top_p': float,
                           'stop': str}

    def __init__(self, pretrained: str = "bigcode/starcoder", device_map: str = "auto", bit_precission=16):
        self.timeit()
        self.pretrained: str = pretrained
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.timeit("load tokenizer")
        model = AutoModelForCausalLM.from_pretrained(pretrained, device_map=device_map, **self.get_load_params(bit_precission))
        self.timeit("load model")
        self.pipe: Pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.timeit("load pipeline")
        self.generation_config = GenerationConfig.from_pretrained(pretrained)
        self.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

    def get_load_params(self, float_bits):
        assert float_bits in (32, 16, 8), f"float_bits can only be set to 32, 16 or 8"
        load_params = dict()
        if float_bits == 16:
            load_params = {'torch_dtype': torch.bfloat16}
        if float_bits == 8:
            load_params = {'load_in_8bit': True, 'torch_dtype': torch.float16}
        return load_params

    async def generate(self, query: str, parameters: dict) -> str:
        save_parameters = self.sanitize_parameters(parameters, HfAutoModelCoder.parameter_key_types)
        generation_config_dict = {
            **self.generation_config.to_dict(),
            **save_parameters
        }
        config: GenerationConfig = GenerationConfig.from_dict(generation_config_dict)
        self.timeit()
        json_response: dict = self.pipe(query, generation_config=config)[0]
        generated_text: str = json_response['generated_text']
        self.timeit(f"inference {len(query)}/{len(generated_text)-len(query)}")
        return generated_text

