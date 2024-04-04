import os
from abc import ABC
import numpy as np
import torch
import cv2

from base_api import BaseAPI


class Compose(BaseAPI, ABC):

    def __init__(self,target_h=1280,target_w=768):
        super().__init__()
        self.name = "Compose"
        self.target_h = target_h
        self.target_w = target_w
        self.smallface_up = 150
        self.smallface_down = 75

    def set_params(self,sd_width,sd_height):
        self.target_h = sd_height
        self.target_w = sd_width

    def load(self):
        pass

    def compose(self, data,image,keyp_2d, max_face_id, ratio):
        # image = data['img'].copy()
        # keyp_2d = data['keyp_2d'].copy()
        neck_x, neck_y, _ = keyp_2d[max_face_id][1]
        h, w, c = image.shape
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(image, (new_w, new_h))
        new_neck_x = ratio * neck_x
        new_neck_y = ratio * neck_y

        box_up_x = int(new_neck_x - self.target_w / 2)
        box_up_x = 0 if box_up_x < 0 else box_up_x
        box_up_y = int(new_neck_y - self.target_h / 2)
        box_up_y = 0 if box_up_y < 0 else box_up_y

        box_bot_x = box_up_x + self.target_w
        box_bot_y = box_up_y + self.target_h

        if box_bot_x >= new_w:
            box_bot_x = new_w
            box_up_x = box_bot_x - self.target_w
        if box_bot_y >= new_h:
            box_bot_y = new_h
            box_up_y = box_bot_y - self.target_h

        image_composed = image[box_up_y:box_bot_y, box_up_x:box_bot_x, :]

        keyp_2d[:, :, :2] = keyp_2d[:, :, :2] * ratio
        keyp_2d[:, :, 0] = keyp_2d[:, :, 0] - box_up_x
        keyp_2d[:, :, 1] = keyp_2d[:, :, 1] - box_up_y
        ### verify pose
        for i in range(keyp_2d.shape[0]):
            for j in range(keyp_2d.shape[1]):
                if keyp_2d[i][j][0] < 0 or keyp_2d[i][j][
                        0] >= self.target_w or keyp_2d[i][j][1] < 0 or keyp_2d[
                            i][j][1] >= self.target_h:
                    keyp_2d[i][j][0] = 0
                    keyp_2d[i][j][1] = 0
                    keyp_2d[i][j][2] = 0
        keyp_2d = keyp_2d[np.sum(keyp_2d[:, :, -1]>self.config['keyp2d_thresh'], axis=1)>self.config['min_keyp2d_number']]
        data['new_keyp_2d'] = keyp_2d
        return image_composed

    def run(self, data, need_composion=False):
        '''
        # TODO
        need_composion: False, means just crop and resize
        need_composion: True, means compose the image and then crop and resize
        '''

        # return data['img']
        # image = data['img'].copy()
        image = data['harmony_img'].copy()
        if 'keyp_2d' in data:
            keyp_2d = data['keyp_2d'].copy()
        h, w, c = image.shape
        ### make sure image [h,w]>=[target_h,target_w]
        if w < self.target_w or h < self.target_h:
            scale = max(self.target_h/h,self.target_w/w)
            image = cv2.resize(image,None,fx=scale,fy=scale)
            # image = cv2.resize(image,None,fx=self.target_w/w,fy = self.target_w/w)
            if 'keyp_2d' in data:
                keyp_2d[:,:,:2]=keyp_2d[:,:,:2]*scale
        
        h,w,c = image.shape
        if need_composion:
            ### get max face
            max_face_id = 0
            max_face_area = 0
            max_face_width = 0
            max_face_height = 0
            ratio = 1.0
            for i in range(keyp_2d.shape[0]):
                if keyp_2d[i][17][2] > 0.4 and keyp_2d[i][16][2] > 0.4:
                    face_width = keyp_2d[i][17][0] - keyp_2d[i][16][
                        0]  ###left ear - right ear
                    face_height = face_width * 1.4
                    face_area = abs(face_width * face_height)
                    if face_area > max_face_area:
                        max_face_id = i
                        max_face_area = face_area
                        max_face_width = abs(face_width)
                        max_face_height = abs(face_height)
            ### if max face between[75,150],compose image
            if (max_face_height <= self.smallface_up
                    and max_face_height >= self.smallface_down) or (max_face_width <= self.smallface_up
                                                   and max_face_width >= self.smallface_down):
                ratio = max(self.smallface_up / max_face_width,
                            self.smallface_up / max_face_height)
                composed_img = self.compose(data, image,keyp_2d,max_face_id, ratio)
            else:
                ratio = self.target_w/w
                ### if after resize,max face between[75,150],compose image
                if (max_face_height*ratio<=self.smallface_up and max_face_height*ratio>=self.smallface_down)or (
                            max_face_width * ratio <= self.smallface_up
                            and max_face_width * ratio >= self.smallface_down):
                    ratio_ = max(
                        self.smallface_up / (max_face_height * ratio),
                        self.smallface_up / (max_face_width * ratio))
                    ratio = ratio*ratio_
                    composed_img = self.compose(data, image,keyp_2d,max_face_id, ratio)
                else:
                    # ratio_w = self.target_w / image.shape[1]
                    # image = cv2.resize(image, None,fx = ratio_w,fy = ratio_w)
                    left_up_y = int((image.shape[0]-self.target_h)/2)
                    left_up_x = int((image.shape[1]-self.target_w)/2)
                    composed_img = image[left_up_y:left_up_y+self.target_h,left_up_x:left_up_x+self.target_w,:]
                    # composed_img = image[left_up_y:left_up_y+self.target_h,:,:]
                    keyp_2d[:,:,1] = keyp_2d[:,:,1]-left_up_y
                    keyp_2d[:,:,0] = keyp_2d[:,:,0]-left_up_x

                    for i in range(keyp_2d.shape[0]):
                        for j in range(keyp_2d.shape[1]):
                            if keyp_2d[i][j][0] < 0 or keyp_2d[i][j][
                                    0] >= self.target_w or keyp_2d[i][j][1] < 0 or keyp_2d[
                                        i][j][1] >= self.target_h:
                                keyp_2d[i][j][0] = 0
                                keyp_2d[i][j][1] = 0
                                keyp_2d[i][j][2] = 0
                    keyp_2d = keyp_2d[np.sum(keyp_2d[:, :, -1]>self.config['keyp2d_thresh'], axis=1)>self.config['min_keyp2d_number']]
                    data['new_keyp_2d'] = keyp_2d

            return composed_img
        else:
            # ratio_w = self.target_w / image.shape[1]
            # scale_1 = min(self.target_w/image.shape[1],self.target_h/image.shape[0])
            # image = cv2.resize(image, None,fx = scale_1,fy = scale_1)

            left_up_y = int((image.shape[0]-self.target_h)/2)
            left_up_x = int((image.shape[1]-self.target_w)/2)
            composed_img = image[left_up_y:left_up_y+self.target_h,left_up_x:left_up_x+self.target_w,:]
            
            if 'keyp_2d' in data:
                keyp_2d[:,:,1] = keyp_2d[:,:,1]-left_up_y 
                keyp_2d[:,:,0] = keyp_2d[:,:,0]-left_up_x
                for i in range(keyp_2d.shape[0]):
                    for j in range(keyp_2d.shape[1]):
                        if keyp_2d[i][j][0] < 0 or keyp_2d[i][j][
                                0] >= self.target_w or keyp_2d[i][j][1] < 0 or keyp_2d[
                                    i][j][1] >= self.target_h:
                            keyp_2d[i][j][0] = 0
                            keyp_2d[i][j][1] = 0
                            keyp_2d[i][j][2] = 0
                keyp_2d = keyp_2d[np.sum(keyp_2d[:, :, -1]>self.config['keyp2d_thresh'], axis=1)>self.config['min_keyp2d_number']]
                data['new_keyp_2d'] = keyp_2d
            return composed_img

