from typing import Optional
import pdb
from dataclasses import dataclass
import spacy
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag, LanguageModelError

class ActiveRag(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.init(args)

    def init(self, args):
        self.filter_prob = args.filter_prob
        self.masked_prob = args.masked_prob
        self.max_fianl_answer_length = args.max_fianl_answer_length
        self.nlp = spacy.load("en_core_web_sm")

    @dataclass
    class LLMoutputs:
        text: Optional[str] = None
        tokens_ids: Optional[list[int]] = None
        tokens_prob: Optional[list[float]] = None

    def infer(self, query:str)->tuple[str,dict]:
        pdb.set_trace()
        next_iter_flag = True
        answer_len = 0
        final_generation = ''
        iter_step = 1
        generation_track = {} # TODO: active rag generation track
        generation_track[iter_step] = {'instruction': None, 'retrieval_input': query, 'passages':None, 'generation':None}
        print(f'source question -> {query}')
        while next_iter_flag and answer_len < self.max_fianl_answer_length:
            if iter_step == 1:
                retrieval_input = query
                passages = self.retrieval.search(retrieval_input) # 为什么当初要先检索呢？
            collated_passages = self.collate_passages(passages) # 请问这里的 passages 指的是什么呢？？？？？
            target_instruction = self.find_instruction('active_rag-read', self.task)
            inputs = target_instruction.format_map({'passages': collated_passages, 'query': query})
            inputs = inputs + final_generation 
            # get look_ahead
            outputs = self.llm_inference(inputs) # 第一次就要先进行rag吗？难道不应该是让 llm 直接生成吗？？？？这里有问题啊
            print(f'whole look ahead -> {outputs.text}')
            if len(outputs.text)==0:
                break
            # get first sentence from look_ahead
            look_ahead = self._truncate_text(outputs) # 其实就是需要获得第一个句子的 text，id，probs
            print(f'look ahead -> {look_ahead.text}')
            # mask low prob tokens in look_ahead
            masked_look_ahead = self._mask_lowprob_tokens(look_ahead)
            if len(masked_look_ahead.tokens_ids) > 0:
                # re-retrieve passages based on look_ahead
                print(f'retrieval input/masked look ahead -> { masked_look_ahead.text}')
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
            if len(outputs.text) == 0:
                break
            # get the first sentence from outputs 
            Doc = self.nlp(outputs.text)
            first_sentence = list(Doc.sents)[0].text
            final_generation += ' ' + first_sentence
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
        return final_generation, generation_track
        
    def llm_inference(self, input:str)-> LLMoutputs : 
        '''
        Because of the inconsistency between the VLLM and tokenizer, 
        the current version of Active RAG cannot use the VLLM for inference acceleration. 
        The next version of RAGLab will work on improving this issue.
        '''
        if self.llm_mode == 'HF_Model':
            input_ids = self.tokenizer.encode(input, return_tensors="pt").cuda() # The output of tokenizer need convert to GPU, because tokenize which is stored in CPU is a pure python object.
            instruction_len = input_ids.shape[1]
            outputs = self.llm.generate(input_ids, do_sample = False, max_length = instruction_len + self.generate_maxlength, output_scores=True, return_dict_in_generate=True)
            # get generation tokens id
            tokens_ids = outputs.sequences[0][instruction_len:] 
            text = self.tokenizer.decode(tokens_ids, skip_special_tokens = False)
            # replace special tokens
            if '</s>' in text:
                text =  text.replace("<s> ", "").replace("</s>", "").strip()
            else:
                text =  text.replace("<s> ", "").strip()
            # calculate the probs of each tokens
            tokens_prob = []
            for idx, token_id in enumerate(tokens_ids):
                token_prob = outputs.scores[idx].log_softmax(-1).exp()[0][token_id].item() # `outputs.scores` only records the logits of the generated tokens, so its length is equal to `generation_maxlength`.
                tokens_prob.append(token_prob)
        elif self.llm_mode == 'Openai_api':
            apioutput_list = self.llm.generate(input)
            Apioutputs = apioutput_list[0]
            text = Apioutputs.text
            tokens_ids = Apioutputs.tokens_ids
            tokens_prob = Apioutputs.tokens_prob
        else:
            raise LanguageModelError("Language model must be huggingface or openai api.")
        return self.LLMoutputs(text, list(tokens_ids), tokens_prob)

    def _truncate_text(self, llm_outputs)->LLMoutputs: 
        '''
        '''
        Doc = self.nlp(llm_outputs.text)
        first_sent = list(Doc.sents)[0].text
        first_sent_tokenid = self.tokenizer.encode(first_sent)
        first_sent_len = len(first_sent_tokenid) # 其实最重要的是拿到第一个 tokens 的长度，这部分其实好像是可以不用动的，只需要
        first_sent_prob = llm_outputs.tokens_prob[0:first_sent_len]
        return self.LLMoutputs(first_sent, first_sent_tokenid, first_sent_prob)

    def _mask_lowprob_tokens(self, llm_outputs):
        '''
        raglab rerpoduce the Masked sentences as implicit queries in active rag algorithm(https://arxiv.org/abs/2305.06983)
        '''
        masked_text = ''
        masked_tokens_ids = []
        masked_tokens_prob = []
        filered_prob = [prob for prob in llm_outputs.tokens_prob if prob < self.filter_prob]
        if len(filered_prob)>0:
            for token_id, token_prob in zip(llm_outputs.tokens_ids, llm_outputs.tokens_prob):
                if token_prob > self.masked_prob:
                    masked_tokens_ids.append(token_id)
                    masked_tokens_prob.append(token_prob)
            masked_text = self.tokenizer.decode(masked_tokens_ids)
        # end of if
        if '</s>' in masked_text:
            masked_text =  masked_text.replace("<s> ", "").replace("</s>", "").strip()
        else:
            masked_text =  masked_text.replace("<s> ", "").strip()
        return self.LLMoutputs(masked_text, masked_tokens_ids, masked_tokens_prob)