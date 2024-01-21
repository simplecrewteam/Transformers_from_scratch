import torch
import torch.nn as nn
from typing import Any



from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self,data_set, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len): 
        super().__init__()
        
        self.seq_len = seq_len
        self.data_set = data_set
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')]).to(torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')]).to(torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')]).to(torch.int64)

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, index: Any) -> Any:

        src_target_pair = self.data_set[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # This 'encode' will map each word in the sentence to corresponding input ids and ".ids" will give us the corresponding ids of the input sentence 
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Now we have to pad the tokens with the [PAD] to fill up to the sequence length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # In this case we don't have the EOS token so we write 1 here

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens<0:
                raise ValueError('Sentence is too long.')
        
        # Now we have to build 2 tensors one for the encoder and other for the decoder input, but the 3rd we have to build for the target sentnce from the decoder.
        encoder_input = torch.cat(
             [
                  self.sos_token,
                  torch.tensor(enc_input_tokens, dtype=torch.int64),
                  self.eos_token,
                  torch.tensor(enc_num_padding_tokens*[self.pad_token], dtype = torch.int64)
             ]
        )

        decoder_input = torch.cat([
             self.sos_token,
             torch.tensor(dec_input_tokens,dtype=torch.int64),
             torch.tensor(dec_num_padding_tokens*[self.pad_token],dtype=torch.int64)
        ])

        label_output = torch.cat(
             [
                  torch.tensor(dec_input_tokens,dtype= torch.int64),
                  self.eos_token,
                  torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
             ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label_output.size(0) == self.seq_len

        return {
             "encoder_input" : encoder_input, # (Seq_len)
             "decoder_input": decoder_input,  # (Seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
            "decoder_mask" :  (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len),# Here we build a method name causal mask this will give us a matrix of shape (seq_len, seq_len)
            "label": label_output, # (Seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal = 1)
    return mask == 0
        
