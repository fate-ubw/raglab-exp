
def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction: 
            return 1
    return 0