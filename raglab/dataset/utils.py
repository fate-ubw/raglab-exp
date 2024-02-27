import os
from datetime import datetime
import jsonlines
import json
from ruamel.yaml import YAML

def load_jsonlines(file:str)-> list[dict]:
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst 

def get_dataset(task: str, output_dir:str, llm_path: str, eval_datapath: str, rag: str = "selfrag") -> object:
    from raglab.dataset.PopQA import PopQA
    from raglab.dataset.PubHealth import PubHealth
    from raglab.dataset.ArcChallenge import ArcChallenge
    from raglab.dataset.TriviaQA import TriviaQA
    from raglab.dataset.HotpotQA import HotpotQA
    from raglab.dataset.QReCC import QReCC
    from raglab.dataset.SQuAD import SQuAD
    from raglab.dataset.ASQA import ASQA
    from raglab.dataset.Factscore import Factscore
    from raglab.dataset.base_dataset.MultiChoiceQA import MultiChoiceQA
    from raglab.dataset.base_dataset.QA import QA
    if 'PopQA' == task:
        EvalData = PopQA(output_dir, llm_path, eval_datapath)
    elif 'PubHealth' == task:
        EvalData = PubHealth(output_dir, llm_path, eval_datapath)
    elif 'ArcChallenge' == task:
        EvalData = ArcChallenge(output_dir, llm_path, eval_datapath)
    elif 'TriviaQA' == task:
        EvalData = TriviaQA(output_dir, llm_path, eval_datapath)
    elif 'ASQA' == task:
        EvalData = ASQA(output_dir, llm_path, eval_datapath)
    elif 'Factscore' == task:
        EvalData = Factscore(output_dir, llm_path, eval_datapath)
    elif 'HotpotQA' == task:
        EvalData = HotpotQA(output_dir, llm_path, eval_datapath, rag = "dsp")
    elif 'QReCC' == task:
        EvalData = QReCC(output_dir, llm_path, eval_datapath, rag = "dsp")
    elif 'SQuAD' == task:
        EvalData = SQuAD(output_dir, llm_path, eval_datapath, rag = "dsp")
    return EvalData

