from ..utils import STATE_DICT, MODEL_INFO, BUILD_DIR
import torch
import os
from . import IM2LatexModel
from transformers import AutoTokenizer

def load_model(
    state_dict=os.path.join(BUILD_DIR, STATE_DICT),
    model_info=os.path.join(BUILD_DIR, MODEL_INFO),
    tokenizer=os.path.join(BUILD_DIR, 'tokenizer')
):
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(model_info, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    model = IM2LatexModel(
        len(tokenizer),
        model_info['embed_size'],
        model_info['hidden_size'],
        model_info['num_layers'],
        eos_index = tokenizer.eos_token_id
    )

    model.load_state_dict(
        torch.load(state_dict, weights_only=False, map_location=torch.device('cpu'))
    )
    model.to(device)
    return model, tokenizer
