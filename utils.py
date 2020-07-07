import os
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import spacy
from spacy.tokens import Doc

from transformers import BertTokenizer, BertModel
import torch


nlp = spacy.load("en_core_web_md")

import warnings
warnings.filterwarnings("ignore")


class Processor(object):
    def __init__(self, datadir="/home/swift/BERT-GAT-NER/data/conll2003"):
        self._datadir = datadir
        self._sentences = {}    # sentences {"train": [["I", "love", "you"], ["eat", "apple"]]} 
        self._lables = {}    # labels {"train": [["B", "I", "O"], ["B", "O"]]}
        self._vocab = set()
    

    def load_data(self):
        datasets = ["train", "dev", "test"]
        for setname in datasets:
            self._sentences[setname] = []
            self._lables[setname] = []
            datapath = os.path.join(self._datadir, "{}.txt".format(setname))
            with open(datapath, "r", encoding="utf8") as fr:
                lines = fr.read().split("\n\n")[:-1]
                for line in lines:
                    word_label_arr = line.split("\n")
                    sentence = [ item.split()[0] for item in word_label_arr ]
                    label = [ item.split()[1] for item in word_label_arr ]

                    self._vocab |= set(sentence)
                    self._sentences[setname].append(sentence)
                    self._lables[setname].append(label)


    def save_sen_label_adj(self, datadir="/home/swift/BERT-GAT-NER/var"):
        """
        save sentence | label | adjacent matrix
        """
        datasets = ["train", "dev", "test"]
        for setname in datasets:
            setdir = os.path.join(datadir, setname)
            for idx, (sentence, label) in enumerate(zip(self._sentences[setname], self._lables[setname])):
                with open(os.path.join(setdir, "{}_s.txt".format(idx)), "w", encoding="utf-8") as fr:
                    fr.write("\t".join(sentence))
                
                with open(os.path.join(setdir, "{}_l.txt".format(idx)), "w", encoding="utf-8") as fr:
                    fr.write("\t".join(label))
        print("===============Done !=================")
        



if __name__ == "__main__":
    processor = Processor()
    processor.load_data()
    processor.save_sen_label_adj()

    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # sentences = processor._sentences
    # max_length = 0
    # for l in sentences:
    #     s = " ".join(l)
    #     ts = tokenizer.tokenize(s)
    #     max_length = max(max_length, len(ts))
    # print(max_length)

