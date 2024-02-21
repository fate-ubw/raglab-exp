import os
import jsonlines
from raglab.dataset.PopQA import  PopQA
from datetime import datetime


class Factscore(PopQA):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)

    def record_result(self, eval_data, final_prediction_with_citation, catation_docs, response_id, generation_track, inference_results):
        postprocessed_result = final_prediction_with_citation[response_id]
        inference_results.append({"input": eval_data["input"], "output": postprocessed_result, "topic": eval_data["topic"],
                        "cat": eval_data["cat"], "intermediate": generation_track["original_splitted_sentences"][response_id]}) 
        return inference_results