import json


def change_keys(read_path: str, store_path: str) -> None:
    """
    这个方法是为了将原数据中Answer关键词换成answers, Question关键词换成question
    """
    with open(read_path, 'r', encoding='utf-8') as file:
        data = json.load(file)["data"]

    for item in data:
        if 'Answer' in item:
            item['answers'] = item.pop('Answer')
        if 'Question' in item:
            item['question'] = item.pop('Question')

    with open(store_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    read_path = "/home/zxw/raglab-exp/data/eval_dataset/2-eval_data/QReCC/qrecc_test.json"
    store_path = "/home/zxw/raglab-exp/data/eval_dataset/2-eval_data/qrecc_test_processed.json"
    change_keys(read_path, store_path)
    

