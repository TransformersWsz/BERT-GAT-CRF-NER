import glob
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer
from transformers import BertModel

import warnings
warnings.filterwarnings("ignore")


max_length = 175    # including [CLS] and [SEP]
tags = ["PAD", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O", "X", "CLS", "SEP"]
tag2idx = { tag:idx for idx, tag in enumerate(tags) }
BATCH_SIZE = 32
EPOCH = 10


class BGCNDataset(Dataset):
    def __init__(self, dir="/home/swift/BERT-GAT-NER/var/train"):
        self.X_files = sorted(glob.glob(os.path.join(dir, "*_s.txt")))
        self.Y_files = sorted(glob.glob(os.path.join(dir, "*_l.txt")))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        with open(self.X_files[idx], "r", encoding="utf-8") as fr:
            s = fr.readline().split("\t")
        
        with open(self.Y_files[idx], "r", encoding="utf-8") as fr:
            l = fr.readline().split("\t")

        ntokens = ["[CLS]"]
        label_ids = [tag2idx["CLS"]]
        for word, label in zip(s, l):    # 遍历每个单词
            tokens = self.tokenizer.tokenize(word)    # 一个单词可能会被分成多个token
            ntokens.extend(tokens)
            for i, _ in enumerate(tokens):
                label_ids.append(tag2idx[label] if i==0 else tag2idx["X"])
        ntokens.append("[SEP]")
        label_ids.append(tag2idx["SEP"])
        

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)
        segment_ids = [0] * max_length

        rest_pad = [0] * (max_length - len(input_ids))    # pad成max_length
        input_ids.extend(rest_pad)
        mask.extend(rest_pad)
        label_ids.extend(rest_pad)

        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "label_ids": label_ids
        }


def collate_fn(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []
    for example in batch:
        input_ids.append(example["input_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        label_ids.append(example["label_ids"])
    
    return {
        "x": {
            "input_ids": torch.tensor(input_ids).cuda(),
            "token_type_ids": torch.tensor(token_type_ids).cuda(),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).cuda()
        },
        "y": torch.tensor(label_ids).cuda()
    }


class BertGatCRFNerModel(nn.Module):  
    
    def __init__(self, hidden_dim=768, dropout=0.2, tag2idx=tag2idx):
        super(BertGatCRFNerModel, self).__init__()
        self.tag_size = len(tag2idx)
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)
    

    def forward(self, x):
        x = self.bert(**x)[0]
        x = self.dropout(x)
        x = self.hidden2tag(x)
        return x


if __name__ == "__main__":
    train_dataset = BGCNDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = BertGatCRFNerModel()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.8)


    model.train()
    for epoch in range(EPOCH):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = batch["x"]
            y = batch["y"]

            loss = model.log_likelihood(x, y)
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print("EPOCH: {} Step: {} Loss: {}".format(epoch+1, i+1, loss.data))

        # scheduler.step()
    print("================== train done! ================")

