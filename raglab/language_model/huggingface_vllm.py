from typing import Union
import numpy as np
from vllm import LLM, SamplingParams
from raglab.language_model.base_lm import BaseLM
import pdb

class HF_VLLM(BaseLM):
    def __init__(self, args):
        super().__init__(args)
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.generation_stop = args.generation_stop

    def load_model(self):
        self.llm = LLM(model=self.llm_path, tokenizer=self.llm_path, dtype=self.dtype)
        if self.generation_stop != '':
            self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, stop=[self.generation_stop], repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        else:
            self.sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, repetition_penalty= 1, max_tokens = self.generate_maxlength, logprobs=32000, skip_special_tokens = False)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, inputs: Union[str,list[str]], sampling_params = None)->list[BaseLM.Outputs]:
        if isinstance(inputs,str):
            inputs = [inputs]
        outputs_list = []
        if sampling_params is None:
            outputs = self.llm.generate(inputs, self.sampling_params)
        else:
            outputs = self.llm.generate(inputs, sampling_params)
        for RequestOutput in outputs:
            Outputs = self.Outputs()
            text = RequestOutput.outputs[0].text
            # replace special tokens
            if '</s>' in text:
                Outputs.text =  text.replace("<s> ", "").replace("</s>", "").strip()
            else:
                Outputs.text =  text.replace("<s> ", "").strip()
            Outputs.tokens_ids = RequestOutput.outputs[0].token_ids
            Outputs.cumulative_logprob = RequestOutput.outputs[0].cumulative_logprob
            Outputs.tokens_num = len(Outputs.tokens_ids)
            # tokens_prob & tokens_logprob
            Outputs.tokens_logprob = [logprob[token_id] for token_id, logprob in zip(Outputs.tokens_ids, RequestOutput.outputs[0].logprobs)]
            Outputs.tokens_prob = np.exp(Outputs.tokens_logprob).tolist()   
            Outputs.logprobs = RequestOutput.outputs[0].logprobs
            outputs_list.append(Outputs)
        # --> end of for loop
        return outputs_list
