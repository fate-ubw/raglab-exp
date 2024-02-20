import sys
import os
import dspy
from raglab.dataset.base_dataset import QA
from raglab.dataset.utils import get_dataset
from raglab.rag.infer_alg.naive_rag.naiverag import NaiveRag
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join('./cache')

class dsp(NaiveRag):
    def __init(self, args):
        self.args = args

if __name__ == "__main__":
    turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key="sk-tFi5dr7s6tfZM9IA99570920Ea464869A88a3aB77128800b", api_base="https://api.aigcbest.top/v1")
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
    from dspy.datasets import HotPotQA

    # Load the dataset.
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

    # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]

    len(trainset), len(devset)