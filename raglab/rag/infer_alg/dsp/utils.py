import dspy
from dsp.utils import deduplicate


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def gold_passages_retrieved(example, pred, trace=None):
    gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
    found_titles = set(map(dspy.evaluate.normalize_text, [c.split(' | ')[0] for c in pred.context]))

    return gold_titles.issubset(found_titles)


# class SimplifiedBaleen(dspy.Module):
#     def __init__(self, passages_per_hop=3, max_hops=2):
#         super().__init__()

#         self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery, temperature=0.7) for _ in range(max_hops)]
#         self.retrieve = dspy.Retrieve(k=passages_per_hop)
#         self.generate_answer = dspy.ChainOfThought(GenerateAnswer, temperature=0.7)
#         self.max_hops = max_hops
    
#     def forward(self, question):
#         context = []
        
#         for hop in range(self.max_hops):
#             query = self.generate_query[hop](context=context, question=question).query
#             passages = self.retrieve(query).passages
#             context = deduplicate(context + passages)

#         pred = self.generate_answer(context=context, question=question)
#         return dspy.Prediction(context=context, answer=pred.answer)
    

class SimplifiedBaleen(dspy.Module):
    def __init__(self, retrieve, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery, temperature=0.7) for _ in range(max_hops)]
        self.retrieve = retrieve
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer, temperature=0.7)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve.search(query)
            passages = [passages[rank]['content'] for rank in sorted(passages)]
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    

def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True

