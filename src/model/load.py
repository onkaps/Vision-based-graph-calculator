from ..utils import STATE_DICT, MODEL_INFO, BUILD_DIR
import torch
import os
from . import IM2LatexModel


def load_model(
    state_dict=os.path.join(BUILD_DIR, STATE_DICT),
    model_info=os.path.join(BUILD_DIR, MODEL_INFO)
):
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(model_info, weights_only=False)

    model = IM2LatexModel(
        model_info['vocab_size'],
        model_info['embed_size'],
        model_info['hidden_size'],
        model_info['num_layers']
    )

    model.load_state_dict(
        torch.load(state_dict, weights_only=False, map_location=torch.device('cpu'))
    )
    model.to(device)
    return model
