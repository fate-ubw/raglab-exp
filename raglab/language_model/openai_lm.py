import os
import openai
import sys
import time
import logging
from pprint import pprint
from typing import Optional
import numpy as np
from dataclasses import dataclass
import tiktoken
import pdb

class OpenaiModel:
    def __init__(self,args):
        self.llm_name = args.llm_name
        self.api_key_path = args.api_key_path
        self.api_base = args.api_base
        self.temperature = args.temperature
        self.generation_stop = args.generation_stop
        if self.generation_stop == '':
            self.generation_stop = None
        self.generate_maxlength = args.generate_maxlength
        self.top_p = args.top_p
        self.api_logprobs = args.api_logprobs
        self.api_top_logprobs = args.api_top_logprobs

    @dataclass
    class Apioutputs:
        '''
        Apioutputs unify all kinds of output of openai api
        '''
        text: Optional[str] = None
        tokens_id: Optional[list[int]] = None
        completion_tokens: Optional[int] = None
        prompt_tokens: Optional[int] = None
        tokens:Optional[list[str]] = None
        tokens_prob: Optional[list[float]]= None
        tokens_logprob: Optional[list[float]]= None
        
        def __repr__(self):
            # TODO 但是不知道为什么没有被调用，直接打印也不行，非常的奇怪
            return (f"Apioutputs(text={self.text}, "
                f"tokens_id={self.tokens_id}, "
                f"tokens_prob={self.tokens_prob}, "
                f"completion_tokens={self.completion_tokens}, "
                f"prompt_tokens={self.prompt_tokens})")

    def load_model(self):
        # load api key
        key_path = self.api_key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        openai.api_key = api_key.strip()
        openai.api_base = self.api_base
        print(f'current api base: {openai.api_base}')
        print(f"Current API: {api_key.strip()}")
        self.tokenizer = tiktoken.encoding_for_model(self.llm_name)

    def generate(self,input:str)-> Apioutputs:
        '''
        Current version of OpenaiModel batch inference was not implemented
        '''
        for i in range(1,11): # max time of recall is 10 times
            print(f'The {i}-th API call')
            message = [{"role": "user", "content": input}] #TODO logprobs 如何设置 
            response = self.call_ChatGPT(message, model_name=self.llm_name, max_len=self.generate_maxlength, temp=self.temperature, top_p=self.top_p, stop = self.generation_stop, logprobs=self.api_logprobs, top_logprobs=self.api_top_logprobs)
            # collate Apioutputs
            self.Apioutputs.text = response["choices"][0]["message"]["content"]
            self.Apioutputs.completion_tokens = response["usage"]["completion_tokens"]
            self.Apioutputs.prompt_tokens = response["usage"]["prompt_tokens"]
            if self.api_logprobs is True and response["choices"][0]['logprobs'] is not None:
                # some situation api will not return logprobs althought the setting is right
                self.Apioutputs.tokens = [content['token'] for content in response["choices"][0]['logprobs']['content']]
                self.Apioutputs.tokens_logprob = [content['logprob'] for content in response["choices"][0]['logprobs']['content']]
                self.Apioutputs.tokens_prob = np.exp(self.Apioutputs.tokens_logprob).tolist()
                self.Apioutputs.tokens_id = self.tokenizer.encode(self.Apioutputs.text)
                print(f'API call success')
                break
        return self.Apioutputs

    def call_ChatGPT(self,message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, top_p = 1.0, stop = None, logprobs = False, top_logprobs = None, verbose=False):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = openai.ChatCompletion.create(model=model_name,
                                                        messages=message,
                                                        max_tokens=max_len,
                                                        temperature=temp,
                                                        top_p = top_p,
                                                        stop = stop,
                                                        logprobs = logprobs,
                                                        top_logprobs = top_logprobs)
                received = True
            except:
                num_rate_errors += 1
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    assert False
                logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
                time.sleep(np.power(2, num_rate_errors))
        return response

