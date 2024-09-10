from tqdm import tqdm
import pandas as pd 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(neg_pos_df: pd.DataFrame,
                      list_ques: list,
                      tokenizer_from_pretrained: str,
                      num_neg: int = 7,
                      num_pos: int = 8,
                      q_len: int = 64,
                      doc_len: int = 256,
                      batch_size: int = 256,
                      shuffle: bool = True
                     ):
    
    list_doc_pos, list_doc_neg = [], []
    new_lq = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_from_pretrained)
    for ques in tqdm(list_ques,
                          desc="Question ..."):
        curr_df = neg_pos_df[neg_pos_df["question"] == ques]
        pos_df = curr_df[curr_df["label"] == 1].sort_values(by='score_bm25', ascending=False)
        neg_df = curr_df[curr_df["label"] == 0].sort_values(by='score_bm25', ascending=False)
        pos_docs = pos_df["doc"]
        negss = neg_df["doc"].iloc[0:num_neg].tolist()
        if num_neg > 0:
            for idx, pos_d in enumerate(pos_docs):
                if idx == num_pos:
                    break
                new_lq.append(ques)
                list_doc_pos.append(pos_d)
                list_doc_neg.extend(negss)
        else:
            for idx, pos_d in enumerate(pos_docs):
                if idx == num_pos:
                    break
                list_doc_pos.append(pos_d)
    
    ques_token = tokenizer(new_lq, return_tensors='pt', 
                                max_length=q_len, padding='max_length', truncation=True)
    pos_token = tokenizer(list_doc_pos, return_tensors='pt',
                             max_length=doc_len, padding='max_length', truncation=True)
    if num_neg > 0:
        neg_token = tokenizer(list_doc_neg, return_tensors='pt',
                             max_length=doc_len, padding='max_length', truncation=True)
        neg_ids = neg_token["input_ids"].view(-1, num_neg, doc_len)
        neg_attn  = neg_token["attention_mask"].view(-1, num_neg, doc_len)
        dataset = TensorDataset(ques_token["input_ids"], ques_token["attention_mask"],
                               pos_token["input_ids"], pos_token["attention_mask"],
                               neg_ids, neg_attn)
    else:
        dataset = TensorDataset(ques_token["input_ids"], ques_token["attention_mask"],
                               pos_token["input_ids"], pos_token["attention_mask"])
        
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader

