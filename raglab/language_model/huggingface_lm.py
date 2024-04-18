from typing import  Union
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from raglab.language_model.base_lm import BaseLM
import pdb

class HF_Model(BaseLM):
    def __init__(self,args):
        super().__init__(args)
        self.llm_path = args.llm_path
        self.dtype = args.dtype
        self.generation_stop = args.generation_stop

    def load_model(self):
        if self.dtype == 'half' or self.dtype == 'float16':
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto", torch_dtype=torch.float16)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, skip_special_tokens=False, padding_side="left")

    def generate(self, inputs: Union[str,list[str]])->list[BaseLM.Outputs]:
        if isinstance(inputs,str):
            inputs = [inputs]
        outputs_list = []
        for prompt in tqdm(inputs, desc="Generating outputs"):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            instruction_len = input_ids.shape[1]
            hf_outputs = self.llm.generate(
                input_ids=input_ids,
                do_sample=False,
                max_length=input_ids.shape[1] + self.generate_maxlength,
                output_scores=True,
                return_dict_in_generate=True
            )
            Outputs = self.Outputs()
            Outputs.tokens_ids = hf_outputs.sequences[0][instruction_len:].tolist()
            Outputs.tokens_num = len(Outputs.tokens_ids)
            text = self.tokenizer.decode(Outputs.tokens_ids, skip_special_tokens = False)
            # replace special tokens
            if '</s>' in text:
                Outputs.text =  text.replace("<s> ", "").replace("</s>", "").strip()
            else:
                Outputs.text =  text.replace("<s> ", "").strip()
            # calculate the probs of each tokens
            tokens_prob = []
            tokens_logprob = []
            logprobs = []
            for idx, token_id in enumerate(Outputs.tokens_ids): # attention the type of token_id is torch.tensor()
                token_logprob = hf_outputs.scores[idx].log_softmax(-1)[0][token_id].item()
                token_prob = hf_outputs.scores[idx].log_softmax(-1).exp()[0][token_id].item() # `outputs.scores` only records the logits of the generated tokens, so its length is equal to `generation_maxlength`.
                logprob_dict = {int(i):float(logprob) for i, logprob in enumerate(hf_outputs.scores[idx].log_softmax(-1)[0].tolist())}
                tokens_prob.append(token_prob)
                tokens_logprob.append(token_logprob)
                logprobs.append(logprob_dict)
            Outputs.tokens_logprob = tokens_logprob
            Outputs.tokens_prob = tokens_prob
            Outputs.cumulative_logprob = float(np.prod(Outputs.tokens_prob) / max(len(Outputs.tokens_prob), 1))
            Outputs.logprobs = logprobs
            outputs_list.append(Outputs)
        # --> end of for loop
        return outputs_list