import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = BertModel.from_pretrained('bert-base-cased')
# print(tokenizer(["I love you", "He is playing games"]))
print(tokenizer.tokenize("Wsz is playing games"))
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
# print(last_hidden_states)