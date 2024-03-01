from raglab.dataset.utils import get_dataset
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
import raglab.rag.infer_alg.dsp.utils as dsp_utils
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
import dspy.evaluate.metrics as Metrics
import sys
import os
repo_path = './raglab/rag/infer_alg/dsp'
if repo_path not in sys.path:
    sys.path.append(repo_path)
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(repo_path, 'cache')
import dspy

proxy = "http://100.124.78.167:3389"
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
os.environ['all_proxy'] = proxy
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

class dsp(NaiveRag):
    def __init__(self, args):
        
        self.args = args 

        # RAG method
        self.rag = args.rag

        # dataset
        self.task = args.task
        self.eval_datapath = args.eval_datapath

        # language model
        self.model_model = args.model_mode # this parameter control HFModel or Openai
        self.llm_path = args.llm_path
        self.llm_api = args.llm_api
        self.api_key = args.api_key
        self.api_base = args.api_base
        # generate parameters
        self.generate_maxlength = args.generate_maxlength
        self.temperature = args.temperature
        self.output_dir = args.output_dir

        # # retrieval model
        # self.retrieval_url = args.retrieval_url
        # # retrieval parameters
        # self.passages_per_hop = args.passages_per_hop

        # retrieval model
        self.retrieval_name = args.retrieval_name
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
        # retrieval parameters
        self.nbits = args.nbits
        self.num_gpu = args.num_gpu
        # self.doc_maxlen = args.doc_maxlen
        self.n_docs = args.n_docs

        # dsp parameters
        self.inference_CoT = args.inference_CoT
        self.signature_retrieval = args.signature_retrieval
        self.max_hops = args.max_hops
        self.eval_threads = args.eval_threads

        # evaluate parameters
        self.mrtrics = args.metrics
        
        self.lm = self.load_llm()
        self.rm = self.load_rm()
        dspy.settings.configure(lm=self.lm, rm=self.rm)


    def inference(self, query='', mode='interact'):
        assert mode in ['interact', 'evaluation']
        generator = dspy.ChainOfThought(dsp_utils.BasicQA, temperature=self.temperature)

        if self.signature_retrieval:
            dataset = get_dataset(self.task, self.output_dir, self.llm_path, self.eval_datapath, self.rag).load_dataset()
            trainset = [x.with_inputs('question') for x in dataset.train]
            print(f"\nLength of traing dataset is {len(trainset)}\n")
            print(trainset)
            teleprompter = BootstrapFewShot(metric=dsp_utils.validate_context_and_answer_and_hops)
            compiled_rag = teleprompter.compile(dsp_utils.SimplifiedBaleen(retrieve=self.rm), teacher=dsp_utils.SimplifiedBaleen(retrieve=self.rm), trainset=trainset)
            if 'interact' == mode:
                compiled_rag(query)
                print(self.lm.inspect_history(n=3))
            elif 'evaluation' == mode:
                assert self.signature_retrieval is True
                devset = [x.with_inputs('question') for x in dataset.dev]
                print(f"Length of dev dataset is {len(trainset)}")
                evaluator = Evaluate(devset=devset, num_threads=self.eval_threads)
                metric = dspy.evaluate.answer_exact_match
                return evaluator(compiled_rag, metric=metric)
        else:
            print(query)
            pred = generator(question=query)
            print(f"Question: \n{query} \n\n")
            print(f"Thought: \n{pred.rationale} \n\n")
            print(f"Predicted Answer: \n{pred.answer} \n\n")
            return pred.answer
            
    def load_llm(self):
        try:
            if self.model_model == "HFModel":
                return dspy.HFModel(model=self.llm_path)
            elif self.model_model == "OpenAI":
                return dspy.OpenAI(model=self.llm_api, api_key=self.api_key, api_base=self.api_base)
            else:
                raise ValueError(f"Invalid model_mode: {self.model_mode}. Must be 'HFModel' or 'OpenAI'.")
        except ValueError as e:
            print(e)
    
    def load_rm(self):
        return super().setup_retrieval()
    
    # def load_rm(self):
    #     return dspy.ColBERTv2(url=self.retrieval_url)
        
    def get_instruction(self):
        return self.lm.inspect_history(n=1)
            
    def setup_Signature(self):
        if self.signature_retrieval:
            return dsp_utils.GenerateSearchQuery
        else:
            return dsp_utils.BasicQA

    def setup_generator(self):
        if self.inference_CoT:
            print(type(self.setup_Signature()))
            return dspy.ChainOfThought(self.setup_Signature(), temperature=self.temperature)
        else:
            return dspy.Predict(self.setup_Signature(),temperature=self.temperature)
        
        

    # def test_for_no_retrieve(self):
    #     from dspy.datasets import HotPotQA
    #     # Load the dataset.
    #     dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    #     # dataset = HotPotQA()
    #     # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    #     trainset = [x.with_inputs('question') for x in dataset.train]
    #     devset = [x.with_inputs('question') for x in dataset.dev]
    #     print(len(trainset), len(devset))
    #     dev_example = devset[18]
    #     generate_answer_with_chain_of_thought = dspy.ChainOfThought(dsp_utils.BasicQA, temperature=0.7)
    #     # Call the predictor on the same input.
    #     pred = generate_answer_with_chain_of_thought(question="How many storeys are in the castle that David Gregory inherited?")
    #     # Print the input, the chain of thought, and the prediction.
    #     print(f"Question: \n{dev_example.question} \n\n")
    #     print(f"Thought: \n{pred.rationale} \n\n")
    #     print(f"Predicted Answer: \n{pred.answer} \n\n")
        
    # def test(self):
    #     # turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key="sk-tFi5dr7s6tfZM9IA99570920Ea464869A88a3aB77128800b", api_base="https://api.aigcbest.top/v1")
    #     # turbo = dspy.HFModel(model="/home/wyd/raglab-exp/model/Llama-2-7b-hf")
    #     # colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    #     dspy.settings.configure(lm=self.lm, rm=self.rm)
    #     from dspy.datasets import HotPotQA
    #     # Load the dataset.
    #     dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    #     os.environ['http_proxy'] = ''
    #     os.environ['https_proxy'] = ''
    #     os.environ['all_proxy'] = ''
    #     # dataset = HotPotQA()
    #     # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    #     trainset = [x.with_inputs('question') for x in dataset.train]
    #     devset = [x.with_inputs('question') for x in dataset.dev]
    #     print(len(trainset), len(devset))
    #     dev_example = devset[18]

    #     def validate_context_and_answer_and_hops(example, pred, trace=None):
    #         if not dspy.evaluate.answer_exact_match(example, pred): return False
    #         if not dspy.evaluate.answer_passage_match(example, pred): return False

    #         hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    #         if max([len(h) for h in hops]) > 100: return False
    #         if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    #         return True
        
    #     class GenerateSearchQuery(dspy.Signature):
    #         """Write a simple search query that will help answer a complex question."""

    #         context = dspy.InputField(desc="may contain relevant facts")
    #         question = dspy.InputField()
    #         query = dspy.OutputField()

    #     class GenerateAnswer(dspy.Signature):
    #         """Answer questions with short factoid answers."""

    #         context = dspy.InputField(desc="may contain relevant facts")
    #         question = dspy.InputField()
    #         answer = dspy.OutputField(desc="often between 1 and 5 words")
        
    #     from dsp.utils import deduplicate

    #     class SimplifiedBaleen(dspy.Module):
    #         def __init__(self, passages_per_hop=3, max_hops=2):
    #             super().__init__()

    #             self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery, temperature=0.7) for _ in range(max_hops)]
    #             self.retrieve = dspy.Retrieve(k=passages_per_hop)
    #             self.generate_answer = dspy.ChainOfThought(GenerateAnswer, temperature=0.7)
    #             self.max_hops = max_hops
            
    #         def forward(self, question):
    #             context = []
                
    #             for hop in range(self.max_hops):
    #                 query = self.generate_query[hop](context=context, question=question).query
    #                 passages = self.retrieve(query).passages
    #                 context = deduplicate(context + passages)

    #             pred = self.generate_answer(context=context, question=question)
    #             return dspy.Prediction(context=context, answer=pred.answer)

        
    #     teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
    #     compiled_baleen = teleprompter.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset)
    #     compiled_baleen("How many storeys are in the castle that David Gregory inherited?")
    #     print(self.lm.inspect_history(n=3))
        

    # def test_retrieve(self):
    #     dspy.settings.configure(rm=dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts'))
    #     print(dspy.Retrieve(k=1)("How many storeys are in the castle that David Gregory inherited?")) 