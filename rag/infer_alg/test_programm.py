from naive_rag import NaiveRag
import argparse
import pdb
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument("--db_path", type = str, help = 'path to preprocessed databset with index')
    parser.add_argument("--eval_datapath", type = str, help = 'path to eval dataset')
    parser.add_argument("--retriever_path", type = str, help = 'path to colbert model')
    parser.add_argument('--generate_maxlength', type = int, help = 'llm generate max length')
    parser.add_argument('--use_vllm', action = "store_true", help = 'llm generate max length')

    args = parser.parse_args()
    pdb.set_trace()
    rag = NaiveRag(args)
    output = rag.llm_inference('hello what your name')
    # 需要注意的是 generate 的代码需要截取后面的回答，
        #     (Pdb) output_text
        # "<s> hello what your name is?\nI'm a 20 year old girl from the Netherlands.\nI'm a 20 year old girl from the Netherlands. I'm a student and I'm studying to become a"
        # "                         is?\nI'm a 20 year old girl from the Netherlands.\nI'm a 20 year old girl from the Netherlands. I'm a student and I'm studying to become a teacher. I love to"
        # 为什么回答的不一样呢？是因为 generate 中 max length 包含了 prompt 的所以长度不同
    print(output)
