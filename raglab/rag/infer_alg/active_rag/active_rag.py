from typing import Optional, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pdb
import pudb
import re
from dataclasses import dataclass
import spacy
from raglab.dataset.utils import get_dataset # load dataset class
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag

class ActiveRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self, args):
        self.filter_prob = args.filter_prob
        self.masked_prob = args.masked_prob
        self.max_fianl_answer_length = args.max_fianl_answer_length

    @dataclass
    class LLMoutputs:
        text: str
        tokens_id: list[int]
        tokens_prob: list[float]
    
    def inference(self, query:Optional[str]=None, mode = 'interact'):
        assert mode in ['interact', 'evaluation']
        self.nlp = spacy.load("en_core_web_sm")
        if 'interact' == mode:
            final_generation = self.active_rag_generation(query)
            return final_generation
        elif 'evaluation' == mode:
            self.EvalData = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath)
            self.eval_dataset = self.EvalData.load_dataset()
            print(f"\n\n{'*' * 20} \nNow, You are evaluating Task: {self.task} with Dataset {self.eval_datapath} \n{'*' * 20}\n\n")
            inference_results = []
            for idx, eval_data in enumerate(tqdm(self.eval_dataset)):
                generation_track = {} #TODO generation 最重要的是记录信息，尤其是中间的信息，但是只有在评测 ASQA 的时候才会需要这个东西
                question = eval_data[self.EvalData.inputStruction.question]
                final_generation = self.active_rag_generation(question)
                inference_results = self.EvalData.record_result(eval_data, final_generation, inference_results)
                eval_result = self.EvalData.eval_acc(inference_results)
                print(f'{self.task} Accuracy in {idx} turn: {eval_result}')
            self.EvalData.save_result(inference_results)
            eval_result = self.EvalData.eval_acc(inference_results)
            print(f'{self.task} Accuracy: {eval_result}')
            return eval_result
        # end of elif

    def active_rag_generation(self, query:str)->str:
        next_iter_flag = True
        answer_len = 0
        final_generation = ''
        iter_step = 0
        generation_track = {}
        generation_track[iter_step] = {'instruction': None, 'retrieval_input': query, 'passages':None, 'generation':None}
        while next_iter_flag and answer_len < self.max_fianl_answer_length:
            if iter_step == 0:
                retrieval_input = query
                passages = self.retrieval.search(retrieval_input)
            collated_passages = self.collate_passages(passages)
            target_instruction = self.find_instruction('active_rag-read', self.task)
            inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
            inputs = inputs + final_generation 
            # get look_ahead
            outputs = self.llm_inference(inputs)
            print(f'whole look ahead -> {outputs.text}')
            # get first sentence from look_ahead
            look_ahead = self.truncate_text(outputs)
            print(f'look ahead -> {look_ahead.text}')
            # mask low prob tokens in look_ahead
            masked_look_ahead = self.mask_lowprob_tokens(look_ahead)
            if len(masked_look_ahead.tokens_id) > 0: 
                # re-retrieve passages based on look_ahead
                print(f'retrieval input/masked look ahead:{ masked_look_ahead.text}')
                # retrieval
                passages = self.retrieval.search(masked_look_ahead.text)
                collated_passages = self.collate_passages(passages)
                target_instruction = self.find_instruction('active_rag-read', self.task)
                inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
                # concatenate instruction + question + least answer
                inputs = inputs + final_generation 
                outputs = self.llm_inference(inputs)
                print(f'final outpus -> {outputs.text}')
            else:
                # If no low prob tokens in look_ahead, look_ahead is the current turn outputs
                outputs = look_ahead

            # get the first sentence from outputs 
            Doc = self.nlp(outputs.text)
            first_sentence = list(Doc.sents)[0].text
            final_generation += first_sentence
            print(f'final generation -> {final_generation}')
            # clculate the len of current generation length
            truncated_outputs_id = self.tokenizer.encode(first_sentence)
            answer_len +=  len(truncated_outputs_id)
            number_of_sents = len(list(Doc.sents))
            # Decide continue or not.If the final_outputs contains more than one sentence, the next_iter_flag will set True
            if number_of_sents > 1:
                next_iter_flag = True
            else:
                next_iter_flag = False
            iter_step += 1
        # end of while
        return final_generation
        
    def llm_inference(self, inputs:str)-> LLMoutputs : 
        '''
        Because of the inconsistency between the VLLM and tokenizer, 
        the current version of Active RAG cannot use the VLLM for inference acceleration. 
        The next version of RAGLab will work on improving this issue.
        '''
        input_ids = self.tokenizer.encode(inputs, return_tensors="pt").cuda() # The output of tokenizer need convert to GPU, because tokenize which is stored in CPU is a pure python object.
        instruction_len = input_ids.shape[1]
        outputs = self.llm.generate(input_ids, do_sample = False, max_length = instruction_len + self.generate_maxlength, output_scores=True, return_dict_in_generate=True)
        # get generation tokens id
        tokens_id = outputs.sequences[0][instruction_len:] 
        text = self.tokenizer.decode(tokens_id, skip_special_tokens = False)
        # replace special tokens
        if '</s>' in text:
            text =  text.replace("<s> ", "").replace("</s>", "").strip()
        else:
            text =  text.replace("<s> ", "").strip()
        # calculate the probs of each tokens
        tokens_prob = []
        for idx, token_id in enumerate(tokens_id):
            token_prob = outputs.scores[idx].log_softmax(-1).exp()[0][token_id].item()
            tokens_prob.append(token_prob)
        return self.LLMoutputs(text, tokens_id.tolist(), tokens_prob)

    def truncate_text(self, llm_outputs)->LLMoutputs: 
        '''
        '''
        Doc = self.nlp(llm_outputs.text)
        first_sent = list(Doc.sents)[0].text
        first_sent_tokenid = self.tokenizer.encode(first_sent)
        first_sent_len = len(first_sent_tokenid)
        first_sent_prob = llm_outputs.tokens_prob[0:first_sent_len]
        return self.LLMoutputs(first_sent, first_sent_tokenid,first_sent_prob)

    def mask_lowprob_tokens(self, llm_outputs):
        '''
        raglab rerpoduce the Masked sentences as implicit queries in active rag algorithm(https://arxiv.org/abs/2305.06983)
        '''
        masked_text = ''
        masked_tokens_id = []
        masked_tokens_prob = []
        filered_prob = [prob for prob in llm_outputs.tokens_prob if prob < self.filter_prob]
        if len(filered_prob)>0:
            for token_id, token_prob in zip(llm_outputs.tokens_id, llm_outputs.tokens_prob):
                if token_prob > self.masked_prob:
                    masked_tokens_id.append(token_id)
                    masked_tokens_prob.append(token_prob)
            masked_text = self.tokenizer.decode(masked_tokens_id)
        # end of if
        if '</s>' in masked_text:
            masked_text =  masked_text.replace("<s> ", "").replace("</s>", "").strip()
        else:
            masked_text =  masked_text.replace("<s> ", "").strip()
        return self.LLMoutputs(masked_text, masked_tokens_id, masked_tokens_prob)