import dspy
from dsp.utils import deduplicate
import pdb

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
    

class SimplifiedBaleen(dspy.Module): # 其实这个就是最终的dsp inference
    def __init__(self, retrieve, max_hops=2):
        super().__init__()
        # 这个其实 dsp 的整个结构，init 定义所需要的modular，forward 会组合这几个 block 得出最后的结果
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery, temperature=0.7) for _ in range(max_hops)]
        self.retrieve = retrieve
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer, temperature=0.7)
        self.max_hops = max_hops
    
    def forward(self, question): #debug 的时候得跳进来，来这里才能看得清楚
        context = []
        for hop in range(self.max_hops): #
            query = self.generate_query[hop](context=context, question=question).query
            # 第一次的时候好像都没有调用 llama 
            # (Pdb) query -> '"number of storeys in castle David Gregory inherited"'
            passages = self.retrieve.search(query) # 检索对应的 passages,这里的 passages 和 colbert 得到的是一样的dict[int, dict['content', 'score']]
            passages = [passages[rank]['content'] for rank in sorted(passages)] # list[str]
            context = deduplicate(context + passages) # 喔喔看来第二次的hotpot 生成之前就会将passages 拼接到 context 当中，这个 deduplicate 是去除重复的 passages
        pred = self.generate_answer(context=context, question=question)
        # (Pdb) pred = Prediction(rationale='Answer: Bob Dylan',answer='Bob Dylan')
        # 这个应该是想要记录最后的结果，这里的 Prediction 继承了 Example 其实就是存储数据的结构，
        return dspy.Prediction(context=context, answer=pred.answer) #这里为什么又定义了一个Prediction呢，但是这里的参数是不同的，得到结果的长度是 4
    

def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True

