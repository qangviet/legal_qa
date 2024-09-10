import argparse
import pandas as pd
from transformers import AutoTokenizer

from .utils import create_dataloader
from bi_encoder.trainner import BiEncoderTrainer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default=None, help="Legal corpus file")
    parser.add_argument("--neg_pos_file", type=str, default=None, help="Negative positive file")
    # parser.add_argument("--train_file", type=str, default=None, help="Train file")
    # parser.add_argument("--test_file", type=str, default=None, help="Test file")
    parser.add_argument("--biencoder_path", type=str, default=None, help="Output directory")    
    parser.add_argument("--final_path", type=str, default=None, help="Output directory")
    parser.add_argument("--BE_tokenizer", type=str, default='vinai/phobert-base-v2', help="Bi-encoder tokenizer")
    parser.add_argument("--BE_checkpoint", type=str, default=None, help="Bi-encoder checkpoint")
    parser.add_argument("--BE_representation", type=str, default=None, help="Bi-encoder representation")
    parser.add_argument("--BE_freeze", type=bool, default=False, help="Freeze Bi-encoder")
    parser.add_argument("--BE_score", type='str', default='dot', help="Bi-encoder score")
    parser.add_argument("--BE_lr", type=float, default=1e-5, help="Bi-encoder learning rate")
    parser.add_argument("--BE_num_epochs", type=int, default=30, help="Bi-encoder number of epochs")
    parser.add_argument("--grad_cache", type=bool, default=False, help="Gradient cache")
    parser.add_argument("--no_hard", type=int, default=0, help="Number of hard negatives")
    parser.add_argument("--q_len", type=int, default=64, help="Max question length")
    parser.add_argument("--doc_len", type=int, default=256, help="Max document length")
    parser.add_argument("--path_weight_mlm", type=str, default=None, help="Path to weight MLM")
    parser.add_argument("--train_batch_size", type=int, default=512, help="Train batch size")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Val batch size")
    parser.add_argument("--q_chunk_size", type=int, default=64, help="Question chunk size gradient cache")
    parser.add_argument("--doc_chunk_size", type=int, default=256, help="Document chunk size gradient cache")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    args = parser.parse_args()

    neg_pos_df = pd.read_csv(args.neg_pos_file)
    list_ques = neg_pos_df['question'].unique().tolist()
    # tokenizer = AutoTokenizer.from_pretrained(args.BE_tokenizer)
    
    train_data = create_dataloader(neg_pos_df, list_ques[0:2400],
                                   tokenizer_from_pretrained=args.BE_tokenizer,
                                   
                                   )
    