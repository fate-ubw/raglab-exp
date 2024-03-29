

def match(prediction:str, ground_truth:list[str])->int:
    for gt in ground_truth:
        if gt in prediction: 
            return 1
    return 0