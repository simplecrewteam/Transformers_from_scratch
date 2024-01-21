import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from torch.utils.tensorboard import SummaryWriter
from model import build_transformer

from config import get_weights_file_path, get_config

from tqdm import tqdm
import warnings

from dataset import BilingualDataset,causal_mask
import torchmetrics

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel #This the tokenizer we use
from tokenizers.trainers import WordLevelTrainer # This is the class which will train the above tokenizer (So this will create the vocabulary given the list of sentences)
from tokenizers.pre_tokenizers import Whitespace # Split the word based on the white space


from pathlib import Path # This is the library which allows you to create absolute paths given relative paths

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    #Pre compute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(src,src_mask)
    
    # Initialize the decoder input with  sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # Build mask for the decoder input
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        
        # Calcultate the output of the decoder
        out  = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)
        
        #Get the next token
        prob = model.project(out[:,-1])
        #Select the token with the max prob
        _, next_word = torch.max(prob,dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(src).fill_(next_word.item()).to(device)],dim=1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)
        

def run_validation(model, val_data_set, tokenizer_src, tokenizer_tgt,max_len,device, print_msg, global_step, writer, num_examples = 2 ):
    model.eval()
    count=0
    
    source_texts = []
    expected = []
    predicted = []
    
    #Size of the console window
    console_width = "80"
    
    with torch.no_grad():
        for batch in val_data_set:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0)  == 1 , "Batch Size Must be 1 for the validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            #Using tokenizer to convert the output of the model back to the text
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-'+console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            
            if count == num_examples:
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        
            
    

## This whole code is used to build the tokenizer
def get_all_sentences(data_set,lang):
    for item in data_set:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, data_set, lang): # This method is  mainly for building the object of Tokenizer Class
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(data_set,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_data_set(config):
    # date_set_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tgt']}",split = 'train')
    date_set_raw = load_dataset('cfilt/iitb-english-hindi',split='train')
    

    #Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config,date_set_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,date_set_raw,config['lang_tgt'])

    # Keep 90% for train and 10% for validation
    train_ds_size = int(0.9*len(date_set_raw))
    val_ds_size = len(date_set_raw)-train_ds_size

    train_ds_raw, val_ds_raw = random_split(date_set_raw,[train_ds_size,val_ds_size])

    train_data_set = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_data_set = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
 
    max_len_src = 0
    max_len_tgt = 0

    for item in date_set_raw:
        # print(tokenizer_src.encode(item['translation'][config['lang_src']]))
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f'Max length of the source sentence : {max_len_src}')
    print(f'Max length of the target sentence : {max_len_tgt}')

    train_dataloader = DataLoader(train_data_set,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_data_set,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
        model = build_transformer(vocab_src_len,vocab_tgt_len, config['seq_len'],config['seq_len'],config['d_model'])
        return model

def train_model(config):
     #Define the device
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     print(f"Using Device : {device}")

     Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

     train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_data_set(config)
     model = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # TensorBoard
     writer  = SummaryWriter(config['experiment_name'])

     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],eps=1e-9)

     initial_epoch = 0
     global_step = 0
     
     if config['preload']:
        model_filename = get_weights_file_path(config,config['preload']) 
        print(f'Preloading file from {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    #Cross Entropy Loss is used as a loss function
     loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

     #Let's finally build the training loop 

     for epoch in range(initial_epoch, config['num_epochs']):
         
         #Batch iterator using tqdm: Shows progress bar
         batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch : {epoch: 02d}')

         for batch in batch_iterator:
             model.train()
             encoder_input = batch['encoder_input'].to(device)      # (Batch, Seq_len)
             decoder_input = batch['decoder_input'].to(device)      # (Batch, Seq_len)
             encoder_mask = batch['encoder_mask'].to(device)      # (Batch, 1, 1, Seq_len)
             decoder_mask = batch['decoder_mask'].to(device)       # (Batch, 1,Seq_len, Seq_len)

             # Run Tensors through transformers
             encoder_output = model.encode(encoder_input,encoder_mask) #(Batch, Seq_len, d_model)
             decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) #(Batch, Seq_len, d_model)
             proj_output = model.project(decoder_output) #(Batch, Seq_len, tgt_vocab_size)
             
             #Now we are going to compare our output with the label so declaring here
             label = batch['label'].to(device) # (Batch, seq_len)

             loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
             batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

             #Log the loss on tensorboard
             writer.add_scalar('train loss', loss.item(), global_step)
             writer.flush()

             # Backpropagate the loss
             loss.backward()

             # Update the Weights
             optimizer.step()
             optimizer.zero_grad()
             

             global_step += 1
        
         run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device, lambda msg: batch_iterator.write(msg), global_step,writer)

             # Save the model at the end of every epoch
         model_filename = get_weights_file_path(config, f'{epoch:02d}')
         torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(), # All the weights of the model
             'optimizer_state_dict': optimizer.state_dict(),
             'global_step': global_step 
         }, model_filename)
         
         
if __name__ == "__main__":
       
       warnings.filterwarnings('ignore')
       config = get_config()
       
       train_model(config)
        




          
        



        

