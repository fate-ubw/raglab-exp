import os
import argparse
import json
import jsonlines
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from tqdm import tqdm

from raglab.retrieval.retrieve import Retrieve

class ContrieverRrtieve(Retrieve)
    def __init__(self, args):
        self.index_dbPath = args.index_dbPath
        self.text_dbPath = args.text_dbPath
        self.retriever_modelPath = args.retriever_modelPath
    
    def setup_retrieval(self):
        pass

    def search(self)
        pass

        
