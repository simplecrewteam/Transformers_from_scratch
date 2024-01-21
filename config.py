
from pathlib import Path

def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 30,
        "lr": 1e-4,
        "seq_len": 2200,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder":"weights",
        "model_filename":"tmodel",
        "preload": None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)