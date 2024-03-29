from raglab.dataset.PubHealth import PubHealth

TASK_INSTRUCTION = "Given four answer candidates, A, B, C and D, choose the best answer choice."

PROMPT_INSTRUCTION = "### Instruction:\n{instruction}\n\n### Response:\n"

class ArcChallenge(PubHealth):
    def __init__(self, output_dir, llm_path, eval_datapath):
        super().__init__(output_dir, llm_path, eval_datapath)
    
    def preprocess(self, eval_data):
        choices = eval_data["choices"]
        answer_labels = {}
        for i in range(len(choices["label"])):
            answer_key = choices["label"][i]
            text = choices["text"][i]
            if answer_key == "1":
                answer_labels["A"] = text
            if answer_key == "2":
                answer_labels["B"] = text
            if answer_key == "3":
                answer_labels["C"] = text
            if answer_key == "4":
                answer_labels["D"] = text
            if answer_key in ["A", "B", "C", "D"]:
                answer_labels[answer_key] = text
        
        if "D" not in answer_labels:
            answer_labels["D"] = ""
        choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
            answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
        # '\nA: Planetary density will decrease.\nB: Planetary years will become longer.\nC: Planetary days will become shorter.\nD: Planetary gravity will become stronger.'
        if "E" in answer_labels:
            choices += "\nE: {}".format(answer_labels["E"])
        self.choices = choices
        eval_data["answers"] = [eval_data["answerKey"]]
        return eval_data

    def get_instruction(self, prompt):
        if len(TASK_INSTRUCTION) > 0:
            prompt = TASK_INSTRUCTION + "\n\n## Input:\n\n" + prompt + self.choices
        prompt_with_instruction = PROMPT_INSTRUCTION.format_map({"instruction": prompt})
        return prompt_with_instruction
