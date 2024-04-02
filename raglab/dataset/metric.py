

def match(prediction:str, ground_truth:list[str])->int:
    #TODO: 这个还是有比较大的问题，因为问题的种类还是非常多的，用字符串判断会出现假阳的情况
    # 解决办法:可以优使用 tokneizer 将全部str 转化为 token 然后看看是否包含这些 key token id 
    # 这里还需要进行 text norm 的操作需要仔细的编写 em 的分数
    for gt in ground_truth:
        if gt in prediction: 
            return 1
    return 0