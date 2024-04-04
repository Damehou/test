import os
from .hed_net import HEDNetwork

def init_hed(checkpoint=None, device='cuda:0'):
    root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
    checkpoint = os.path.join(root_path, checkpoint)
    model = HEDNetwork(model_path=checkpoint)

    model.to(device)
    model.eval()

    return model
