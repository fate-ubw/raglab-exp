import os
import jsonlines
from raglab.dataset.PopQA import  PopQA
from datetime import datetime

class ASQA(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)

    def save_result(self, inference_result: list[dict])-> None: 
        print('storing result....')
        new_results = {"data": inference_result, "args": [], "total_cost": 0.0, "azure_filter_fail": ""}
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        model_name = os.path.basename(self.llm_path)
        input_filename = os.path.basename(self.eval_datapath)
        eval_Dataname = os.path.splitext(input_filename)[0]
        time = datetime.now().strftime('%m%d_%H%M')
        output_name = f'infer_output-{eval_Dataname}-{model_name}-{time}.jsonl'
        output_file = os.path.join(self.output_dir, output_name)
        with jsonlines.open(output_file, 'w') as outfile: 
            outfile.write(new_results)
        print(f'output file path:{output_file}')
        print('success!')

    def record_result(self, eval_data, final_prediction_with_citation, catation_docs, response_id, generation_track, inference_results):
        eval_data["output"] = final_prediction_with_citation[response_id]
        eval_data["docs"] = catation_docs[response_id] # list[dict]
        if "original_splitted_sentences" in generation_track:
            eval_data['intermediate'] = generation_track['original_splitted_sentences'][response_id]
        inference_results.append(eval_data)
        return inference_results