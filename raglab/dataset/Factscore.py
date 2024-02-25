import os
import jsonlines
from raglab.dataset.PopQA import  PopQA
from datetime import datetime

class InputStruction:
    qeustion:str
    answer:str
    topic:str

class OutputStruction:
    question:str
    output:str
    topic:str

input_strction = {'qeustion': 'query','answer':"answers"}

class Factscore(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
        self.set_data_struction()

    # @abstruct 这个类是必须重写的
    def set_data_struction(self):
        # if task == 'PopQA':
        self.inputStrction = InputStruction
        self.inputStrction.question = 'query'
        self.inputStrction.answer = 'answer'
        self.outputStrction = OutputStruction
        self.outputStruction.question = 'quesiton'
        self.outputStruction.answer = 'output'

    def record_result(self, eval_data, final_prediction_with_citation, catation_docs, response_id, generation_track, inference_results):
        postprocessed_result = final_prediction_with_citation[response_id]
        inference_results.append({"input": eval_data["input"], "output": postprocessed_result, "topic": eval_data["topic"],
                        "cat": eval_data["cat"], "intermediate": generation_track["original_splitted_sentences"][response_id]}) 
        return inference_results

    def save_result(self, inference_result: list[dict])-> None: 
        print('storing result....')
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        model_name = os.path.basename(self.llm_path)
        input_filename = os.path.basename(self.eval_datapath)
        eval_Dataname = os.path.splitext(input_filename)[0]
        time = datetime.now().strftime('%m%d_%H%M')
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl'
        output_file = os.path.join(self.output_dir, output_name)
        
        with jsonlines.open(output_file, 'w') as outfile: 
            outfile.write_all(inference_result)
        print(f'output file path:{output_file}')
        print('success!')