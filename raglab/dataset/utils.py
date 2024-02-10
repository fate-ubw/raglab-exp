import os
from datetime import datetime
import jsonlines
import json
from ruamel.yaml import YAML

def load_jsonlines(file): # 这个可以当做公共的 utils 
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst