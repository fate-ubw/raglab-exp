import pdb
from raglab.dataset.PubHealth import PubHealth

class InputStruction:
    question:str
    answer:str
    choices:str

class OutputStruction:
    question:str
    answer:str
    generation:str

class MMLU(PubHealth):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)

    def set_data_struction(self):
        '''
        The goal of constructing InputStruction and OutputStruction is to achieve the separation of algorithm logic and data, 
        so that users only need to rewrite  set_data_struction() without modifying the algorithm logic.
        '''
        self.inputStruction = InputStruction
        self.inputStruction.question = 'question'
        self.inputStruction.answer = 'answerKey'
        self.inputStruction.choices = 'choices'

        self.outputStruction = OutputStruction
        self.outputStruction.question = 'question'
        self.outputStruction.answer = 'answerKey'
        self.outputStruction.generation = 'generation'
    
    def preprocess(self, eval_data):
        # target: 其实就是将 question 个 answer 拼接起来
        # 其实也很简单就是将 question 和 choices 给拼接起来
        choices = eval_data["choices"]
        postprocess_text = ''
        for answer_text, label in zip(choices['text'], choices['label']):
            postprocess_text += '\n'+ label + ': ' + answer_text
        pdb.set_trace()
        eval_data[self.inputStruction.question] += postprocess_text
        return eval_data