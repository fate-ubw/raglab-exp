from naive_rag import NaiveRag
import argparse
import pdb
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type = int, default = 1, help = 'the number of gpu')
    parser.add_argument('--output_dir', type = str, help = 'the output dir of evaluation')
    parser.add_argument('--mode', type = str, default = 'interact', choices = ['interact', 'evaluation'] help = 'different mode of ingerence')
    parser.add_argument("--llm_path", type = str, help = 'path to llm')
    parser.add_argument("--db_path", type = str, help = 'path to preprocessed databset with index')
    parser.add_argument("--eval_datapath", type = str, help = 'path to eval dataset')
    parser.add_argument("--retriever_path", type = str, help = 'path to colbert model')
    parser.add_argument('--generate_maxlength', type = int, default = 50, help = 'llm generate max length')
    parser.add_argument("--n_docs", type= int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument('--use_vllm', action = "store_true", help = 'llm generate max length')
    parser.add_argument('--doc_maxlen', type = int, default = 300, help = 'the doc max len decided by the wikidata format, here we set 300')
    parser.add_argument('--nbits', type = int, default = 2, help = 'encode each dimension with n bits')
    args = parser.parse_args()
    rag = NaiveRag(args) # 在定义的时候就需要定义清楚
    pdb.set_trace()
    response = rag.inference('tell me how to be a good teacher')
    # (Pdb) outputs = '[Answer]\n                
    # I think the most important thing is to be a good listener. You need to be able to understand what the student is trying to say and what they are trying to achieve. You need to be able to understand what they are trying to say and what they are trying to achieve. You need to be able to understand what they are trying to say and what they are trying to achieve. You need to be able to understand what they are trying to say and what they are trying to achieve. You need to be able to understand what they are trying to say and what they are trying to achieve. You need to be able to understand what they are trying to say and what they are trying to achieve. You need to be able to understand what they are trying to say and what they are trying to achieve. You need to be able to'
    print(output)
