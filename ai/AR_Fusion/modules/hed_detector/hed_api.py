import sys
sys.path.insert(0, 'modules/hed_detector/models')


import os
import cv2
import numpy as np
import math
from einops import rearrange
from abc import ABC
from einops import rearrange

import torch
import torch.nn.functional as F

from base_api import BaseAPI
from modules.hed_detector.models.hed_net import resize_image
from modules.hed_detector.models.api import init_hed

class Image_Hed(BaseAPI, ABC):
    def __init__(self, detect_resolution=1024, device=0):
        super().__init__()
        self.name = "image_hed"
        self.device = torch.device("cuda", device) if torch.cuda.is_available() else "cpu"

        self.detect_resolution = detect_resolution
    
    def load(self):
        self.hed_model = init_hed(self.config["image_hed_ckpt"], device=self.device)

    def run(self, data):
        img = data['composed_img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_hed = self.preprocess(img)
        hed = self.forward(image_hed)
        edge_image = self.postprocess(hed)

        return edge_image
    
    def preprocess(self, img):
        img = img[:, :, ::-1].copy()
        image = resize_image(img, self.detect_resolution)
        image_hed = torch.from_numpy(image).float()
        image_hed = image_hed / 255.0
        image_hed = rearrange(image_hed, 'h w c -> 1 c h w')

        return image_hed

    def forward(self, image_hed):
        image_hed = image_hed.to(self.device)

        with torch.no_grad():
            edge = self.hed_model(image_hed)[0]

        return edge
    
    def postprocess(self, edge):
        edge = edge.cpu().numpy()
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        edge_image = edge[0]

        return edge_image

if __name__ == "__main__":
    model = Image_Hed()

    img_path = '/home/fangli/code/seg/SwinSeg/imgs/x5.jpg'
    # front_path = '../masks/img-3d-9.jpg'

    img = cv2.imread(img_path)

    data = {}
    data['img'] = img
    image = resize_image(img, 1024)
    # front_rgba = cv2.imread(front_path, cv2.IMREAD_UNCHANGED)

    model.load()
    model.run(data)

    # import pdb;pdb.set_trace()
    # depth = np.tile(data['depth_image'][:,:,None], [1,1,3])
    print(data['edge_image'].shape)
    edge = np.tile(data['edge_image'][:,:,None], [1,1,3])
    concat_img = cv2.hconcat([image, edge])
    cv2.imwrite('out2.jpg', data['edge_image'])
    cv2.imwrite('hed2.jpg', concat_img)

