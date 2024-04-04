import os
import cv2
import logging
import bios
import copy
import random
import time
import torch
import numpy as np
from PIL import Image
from skimage import transform as trans

from utils.timer import Timer
from logger import setup_logging
from modules.person_detector.person_detector_api import Person_Detector
from modules.keyp2d_detector.keyp2d_detector_api import Keyp2D_Detector
from modules.face_detector.face_detector_api import Face_Detector, cal_height, topk_bbox
from modules.face_parsing.face_parsing_api import Face_Parsing
from modules.face_attribute.face_attribute_api import Face_Attribute
from modules.midas_depth.img_depth_api import Image_Depth
from modules.person_matte.img_matte_api import Image_Matte
from modules.person_harmony.img_harmony_api import Image_Harmony
from modules.person_attribute.person_attribute_api import Person_Attribute
# from modules.stable_diffusion.sd_interface import SD_Interface
from modules.compose.compose_api import Compose
from modules.hed_detector.hed_api import Image_Hed
from modules.sd_diffuser.sd_api import Stable_Diffusion
from modules.sd_diffuser.facial_restore.utils import (get_face_mask, canny_process, color_transfer, 
                                                      crop_pil_image_with_bbox, combine_images_with_mask,
                                                      paste_back, get_face_box_and_mask
                                                      )
from Exceptions import *

LOG_LEVEL = 1

def resize_image(img, resolution=1024):
    image = img.copy()
    H, W, C = image.shape
    long_side = max(H, W)
    short_side = min(H, W)
    if long_side>= resolution:
        H = float(H)
        W = float(W)
        k = float(resolution) / long_side
        H *= k
        W *= k
        H = int(np.round(H / 8.0)) * 8
        W = int(np.round(W / 8.0)) * 8
        img = cv2.resize(image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    else:
        H = float(H)
        W = float(W)
        k = float(resolution) / long_side
        H *= k
        W *= k
        H = int(np.round(H / 8.0)) * 8
        W = int(np.round(W / 8.0)) * 8
        img = cv2.resize(image, (W, H), interpolation=cv2.INTER_LANCZOS4)

    return img

def merge_img(img, front_img, rect):
    x1,y1,x2,y2 = rect
    h,w,c = front_img.shape
    assert c in [3,4]
    if c == 4:
        front = front_img[:,:,:3]
        alpha = front_img[:,:,3]
    else:
        front = front_img
        alpha = np.ones((h,w)) * 255

    scale_object = cv2.resize(front, (x2-x1,y2-y1))
    scale_alpha = cv2.resize(alpha, (x2-x1,y2-y1)) / 255.

    front_img = np.zeros(img.shape)
    front_img[y1:y2,x1:x2,:] = scale_object
    alpha_img = np.zeros(img.shape[:2])
    alpha_img[y1:y2,x1:x2] = scale_alpha

    alpha_img = np.expand_dims(alpha_img, axis=-1)
    img_merge = front_img * alpha_img + img * (1 - alpha_img)
    img_merge = img_merge.astype(np.uint8)

    return img_merge

def bbox_iou(boxes1, boxes2):
    # 计算矩形框的面积
    area1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
    area2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])

    # 计算交集的坐标
    inter_top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 使用广播机制
    inter_bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)  # 计算交集区域的宽和高
    
    # 计算交集面积
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    # 计算并集面积
    union_area = area1[:, None] + area2 - inter_area

    # 计算IOU
    iou = inter_area / union_area

    return iou

class AIGCGraph(object):
    def __init__(self):
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        config_path = os.path.join(root_path, 'config.yaml')
        self.config = bios.read(config_path)
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        setup_logging(log_dir)
        self.logger = logging.getLogger('AIGC')
        log_levels = {0: logging.WARNING,
                      1: logging.INFO,
                      2: logging.DEBUG
                      }
        self.logger.setLevel(log_levels[LOG_LEVEL])

        # Time manager
        self.main_timer = Timer()
        self.subgraph_timer_dict = {'text2img_graph': Timer(),
                                    'pose_subgraph': Timer(),
                                    'depth_and_hed_subgraph': Timer(),
                                    'pose_and_hed_subgraph': Timer(),
                                    'depth_and_hed_subgraph_special': Timer(),
                                    }
        self.module_timer_dict = {'face_det': Timer(),
                                  'person_det': Timer(),
                                  'key2d_det': Timer(),
                                  'pose_quality': Timer(),
                                  'render_2keyp': Timer(),
                                  'template_match': Timer(),
                                  'attribute_analyze': Timer(),
                                  'seg': Timer(),
                                  'depth': Timer(),
                                  'matte': Timer(),
                                  'harmony': Timer(),
                                  'img2img': Timer(),
                                  'facial_restor': Timer(),
                                  'img2img_pose': Timer(),
                                  'img2img_pose_hed': Timer(),
                                  'img2img_depth_hed': Timer(),
                                  'img2img_depth': Timer(),
                                  'text2img': Timer(),
                                  'hed': Timer(),
                                  'merge': Timer(),
                                  'compose': Timer()
                                  }

    def create_node(self, device):
        self.person_detector = Person_Detector(device=device)
        self.person_detector.load()
        self.key2d_detector = Keyp2D_Detector(device=device)
        self.key2d_detector.load()
        self.face_detector = Face_Detector(device=device)
        self.face_detector.load()
        self.face_parsing = Face_Parsing(device=device)
        self.face_parsing.load()
        self.face_attribute = Face_Attribute(device=device)
        self.face_attribute.load()
        self.image_depth = Image_Depth(device=device)
        self.image_depth.load()
        self.image_matte = Image_Matte(device=device)
        self.image_matte.load()
        self.image_harmony = Image_Harmony(device=device)
        self.image_harmony.load()
        self.image_hed = Image_Hed(device=device)
        self.image_hed.load()
        self.person_attribute = Person_Attribute()
        self.person_attribute.load()
        self.sd_interface = Stable_Diffusion()
        self.sd_interface.load(self.face_detector, self.face_parsing)
        self.compose = Compose()

    def run_depth_and_hed_subgraph(self, job_info, log_name):
        self.subgraph_timer_dict['depth_and_hed_subgraph'].tic(set_start_time=self.main_timer.start_time)
        self.logger.info('[[[run_depth_and_hed_subgraph]]]')
        self.compose.set_params(job_info['sd_width'],job_info['sd_height'])
        composed_img = self.compose.run(job_info, need_composion=False)
        job_info['composed_img'] = composed_img
        depth_img = self.image_depth.run(job_info)
        job_info['depth_img'] = depth_img
        hed_img = self.image_hed.run(job_info)
        job_info['hed_img'] = hed_img

        self.sd_interface.depth_params = self.sd_interface.config['depth_params_hed']
        self.sd_interface.hed_params = self.sd_interface.config['hed_params_depth']
        self.sd_interface.lora_prompt_val = self.sd_interface.config['lora_prompt_val_normal']
        cfg_scale = self.sd_interface.img2img_params['cfg_scale']
        if job_info['style_mode'] in ['lineart','papercut','chinese_anime']:
            denoise_strength = self.sd_interface.img2img_params['denoise_strength'][job_info['style_mode']]
        else:
            denoise_strength = self.sd_interface.img2img_params['denoise_strength']['depth_and_hed_default']

        self.module_timer_dict['img2img_depth_hed'].tic()
        sd_images = self.sd_interface.img2img(prompt='', gender_prompt=job_info['gender_prompt'], init_image=job_info['composed_img'], 
                                            width=job_info['sd_width'], height=job_info['sd_height'],
                                            use_bclip=True, style=job_info['style_mode'],
                                            multiplier=self.sd_interface.lora_prompt_val[job_info['style_mode']],
                                            steps=self.config['img2img_params']['steps'],
                                            cfg_scale=cfg_scale,
                                            denoising_strength=denoise_strength, use_depth=True,
                                            depth_image=job_info['depth_img'],depth_scale=self.sd_interface.depth_params[job_info['style_mode']][0],
                                            use_hed=True, hed_image=job_info['hed_img'],hed_scale=self.sd_interface.hed_params[job_info['style_mode']][0],
                                            ip_scale=self.config['ip_scale'],
                                            seed=self.config['img2img_params']['seed'])
        
        self.module_timer_dict['img2img_depth_hed'].toc()
        job_info['sd_image'] = np.array(sd_images[0])[:, :, ::-1].copy()
        job_info['output_img'] = job_info['sd_image']

        self.subgraph_timer_dict['depth_and_hed_subgraph'].toc()

    # 整个pipeline的入口函数，开始做了人体检测、人脸检测、人体属性检测（年龄和性别），根据人脸数量区分是直接走风格化函数还是人景合成函数
    def run_style_graph(self, job_info):
        job_info['img_ori'] = job_info['img']
        self.main_timer.tic()

        mode = job_info['mode']
        img = job_info['img']

        # 下面3段就是人体检测、人脸检测、属性检测
        self.module_timer_dict['person_det'].tic()
        person_bboxes = self.person_detector.run(job_info)
        self.module_timer_dict['person_det'].toc()
        job_info["person_bbox"] = person_bboxes

        self.module_timer_dict['face_det'].tic()
        face_bboxes, landmarks = self.face_detector.run(img)
        self.module_timer_dict['face_det'].toc()

        detected_objects, out_im = self.person_attribute.run(job_info)
        yolo_face_num = detected_objects.n_faces

        person_flag1 = len(person_bboxes) == 1 and person_bboxes[0][0] == -1.0
        person_flag2 = len(person_bboxes) == 0

        person_num, face_num = 0, 0
        if person_flag1 or person_flag2:
            person_num = 0
        else:
            person_num = person_bboxes.shape[0]
        
        if face_bboxes is None:
            face_num = 0
        else:
            face_num = face_bboxes.shape[0]

        job_info['person_num'] = person_num
        job_info['face_num'] = face_num
        job_info['face_bbox'] = face_bboxes
        job_info['landmarks'] = landmarks

        #这里如果用户图是超过3人的场景我们取人脸面积最大的3个人脸矩形框和关键点信息，方便后续人脸修复，
        # 这里只取3人因为每个人脸修复耗时在2秒左右，这里可以调整数量
        if face_num >= 2:
            n = 3 if face_num >= 3 else 2
            topk_indexes = topk_bbox(face_bboxes, n)
            topk_bboxes, topk_landmarks = face_bboxes[topk_indexes], landmarks[topk_indexes]
            job_info['topk_bbox'] = topk_bboxes
            job_info['topk_landmarks'] = topk_landmarks
        elif face_num == 1:
            x1,y1,x2,y2 = face_bboxes[0]
            face_w, face_h = x2-x1, y2-y1
            face_len = min(face_w, face_h)
            job_info['topk_bbox'] = face_bboxes
            job_info['topk_landmarks'] = landmarks

        job_info['gender_prompt'] = ''
        job_info['fusion'] = False  #这个参数对应是否做了人景合成，后面图对图生成sd_api下img2img函数有用到
        job_info['no_person'] = False

        # xywh = detected_objects.yolo_results.boxes[list(detected_objects.face_to_person_map.keys())].xywh
        # xywhn = detected_objects.yolo_results.boxes[list(detected_objects.face_to_person_map.keys())].xywhn
        # xyxyn = detected_objects.yolo_results.boxes[list(detected_objects.face_to_person_map.keys())].xyxyn
        xyxy = detected_objects.yolo_results.boxes[list(detected_objects.face_to_person_map.keys())].xyxy
        genders = [detected_objects.genders[index] for index in list(detected_objects.face_to_person_map.keys())]
        ages = [detected_objects.ages[index] for index in list(detected_objects.face_to_person_map.keys())]

        #把选取的人脸与前面检测的人脸属性对应上，后面图片生成时会根据性别调整prompt
        job_info['genders'] = [0]
        if face_num >= 1:
            device = xyxy.device
            face_bboxes = torch.tensor(face_bboxes, device=device)
            ious = bbox_iou(face_bboxes, xyxy)
            max_iou_indices = [iou_row.argmax() if iou_row.numel() > 0 and iou_row.max() > 0.5 else -1 for iou_row in ious]
            genders_ind = [0 if gender == 'male' else 1 for gender in genders]
            selected_genders = [genders_ind[index] if index != -1 else 0 for index in max_iou_indices]
            selected_ages = [ages[index] if index != -1 else 20 for index in max_iou_indices]

            if face_num > 1:
                topk_indexes_list = topk_indexes.tolist()
                topk_genders = [selected_genders[i] for i in topk_indexes_list]
                topk_ages = [selected_ages[i] for i in topk_indexes_list]
            else:
                topk_genders = selected_genders
                topk_ages = selected_ages
            job_info['topk_genders'] = topk_genders
            job_info['topk_ages'] = topk_ages
            job_info['gender_prompt'] = self.attribute_analyze(job_info)

        if face_num == 1:
            job_info['face_len'] = face_len
        job_info['face_num'] = face_num

        #这里的循环时是如果原图无人脸但是生成的图片有人脸就再去生成一次
        #人脸数量为1且人脸大小超过50就走人景合成函数分支，其他的就走直接风格化函数分支
        for i in range(2):
            if face_num == 1 and face_len >= 50:
                self.run_fusion_img2img_graph(job_info)
            else:
                self.run_normal_img2img_graph(job_info)
            
            gen_face_num = self.person_attribute.detect(job_info['output_img'])
            if yolo_face_num >= 1 or gen_face_num <= yolo_face_num: break

        total_time = self.main_timer.toc()
        # job_info['infer_time'].append(round(total_time,2))
        self.main_timer.clear()
        self.logger.info(f'{mode} - Time used: {total_time}')

    def attribute_analyze(self, job_info):
        genders = job_info['genders']

        num_male, num_female = 0, 0
        for gender in genders:
            if gender == 0: num_male+=1
            else: num_female += 1

        prompt = ''
        if num_female==0 and num_male!=0:
            prompt = '%s young male in image,'%num_male
        if num_male == 0 and num_female!=0:
            prompt = '%s female in image,'%num_female
        if num_female!=0 and num_male!=0:
            prompt = '%s young male and %s female in image,'%(num_male,num_female)

        return prompt

    #这是直接风格化函数分支
    def run_normal_img2img_graph(self, job_info):
        mode = job_info['mode']
        task_id = job_info['task_id']
        log_name = f'{task_id}-{mode}:'

        img = job_info['img']

        #根据模式选择缩放到不同尺寸
        if mode == 'photo':
            image = resize_image(img, resolution=1152)
        elif mode == 'phone':
            image = resize_image(img)
        elif mode == 'coin':
            image = resize_image(img)
        elif mode == 'puzzle':
            image = resize_image(img, resolution=1088)
        elif mode == 'fridge':
            image = resize_image(img, resolution=800)
        elif mode == 'totebag':
            image = resize_image(img, resolution=1280)
        elif mode == 'hoodie':
            image = resize_image(img, resolution=1280)
        else:
            pass

        new_height, new_width = image.shape[:2]
        job_info['sd_width'] = new_width
        job_info['sd_height'] = new_height
        job_info['img'] = image

        #这是风格化
        job_info['harmony_img'] = job_info['img']
        self.run_depth_and_hed_subgraph(job_info, log_name)

        #这里做人脸修复，判断条件job_info['no_person']是因为需求要保证即使之前定义为异常的图也要去做风格化，
        #会有人脸检测存在一个人脸但是分割matting没有人像的情况会从人景合成分支跳到风格化分支，因为实际没有人脸所以不用做人脸修复
        if job_info['face_num'] >= 1 and  not job_info['no_person']:
            try:
                self.sd_interface.facial_restoration(job_info)
            except:
                print('face detector failed')

        img = job_info['output_img']
        h, w = img.shape[:2]
        offset = 80

        logo = job_info['logo_image']
        logo_h, logo_w = logo.shape[:2]

        width = int(0.7*logo_w)
        height = int(0.7*logo_h)

        offset_x = w - width - offset
        offset_y = h - height - offset
        rect = (offset_x, offset_y, offset_x+width, offset_y+height)

        img = job_info['output_img']
        img_merge = merge_img(img, logo, rect)
        job_info['output_img'] = img_merge

    def run_fusion_img2img_graph(self, job_info):
        img = job_info['img']
        mode = job_info['mode']
        task_id = job_info['task_id']
        log_name = f'{task_id}-{mode}:'

        person_bboxes = job_info['person_bbox']
        x1, y1, x2, y2 = int(person_bboxes[0][0]), int(person_bboxes[0][1]), int(person_bboxes[0][2]), int(person_bboxes[0][3])
        box_width, box_height = x2 - x1, y2 - y1

        pred_alpha, pred_mask = self.image_matte.run(job_info)

        mask = (pred_mask * 255).astype(np.uint8)
        index = np.where(mask > 100)
        try:
            top, bottom = index[0].min(), index[0].max()
            left, right = index[1].min(), index[1].max()
        except:
            #这里就是分割模型判断没有人像跳到直接风格化分支
            # raise NoPersonException
            job_info['no_person'] = True
            self.run_normal_img2img_graph(job_info)
            return
        
        crop_alpha = pred_alpha[top:bottom,left:right]
        crop_mask = pred_mask[top:bottom,left:right,:]
        crop_img = img[top:bottom,left:right,:]

        width, height = right-left, bottom-top

        #这里是因为人像分割个别情况下会分割不准用人像检测框来判断是否准确来兜底异常情况
        ratio_w, ratio_h = width / box_width, height / box_height
        if ratio_h >= 1.5 or ratio_h <= 0.67 or ratio_w >= 1.5 or ratio_w <= 0.67:
            # raise ImageSizeUnqualifiedException
            self.run_normal_img2img_graph(job_info)
            return
        
        #这部分关键点检测和关键点估计是确定原图中的人像是全身像还是半身像
        keyp_2d, rot = self.key2d_detector.run(job_info)
        job_info['keyp_2d'] = keyp_2d

        ori_pose = self.key2d_detector.evaluate_pose_quality(job_info['keyp_2d'], job_info['img'].shape[:2])
        job_info['ori_pose'] = ori_pose

        ori_h, ori_w = img.shape[:2]

        #根据全身还是半身来选择对应的背景图，背景图一部分适合放全身人像一部分适合放半身人像
        if ori_pose:
            if job_info['mode'] == 'totebag':
                back_img = job_info['bgs_whole'][0]
                local = job_info['locs_whole'][0]
            else:
                if ori_h > ori_w:
                    back_img = job_info['bgs_whole'][1]
                    local = job_info['locs_whole'][1]
                else:
                    back_img = job_info['bgs_whole'][0]
                    local = job_info['locs_whole'][0]
        else:
            if job_info['mode'] == 'totebag':
                back_img = job_info['bgs_half'][0]
                local = job_info['locs_half'][0]
            else:
                if ori_h > ori_w:
                    back_img = job_info['bgs_half'][1]
                    local = job_info['locs_half'][1]
                else:
                    back_img = job_info['bgs_half'][0]
                    local = job_info['locs_half'][0]

        h,w,c = back_img.shape

        #这里根据loc_dict字典里的比例确定缩放后的人像高、宽 
        scale_ratio = local[-1]
        new_height = int(h * scale_ratio)
        new_width = int(new_height / height * width)
        if new_width > w:
            new_width = int(w * scale_ratio)
            new_height = int(new_width / width * height)

        #如果缩放后不能放入背景图直接风格化，没遇到
        if new_width > w or new_height > h:
            # raise ImageSizeUnqualifiedException
            self.run_normal_img2img_graph(job_info)
            return

        #先根据分割结果判断人像是否贴近左、右、上的边
        near_top = True if top <= 5 else False
        near_left = True if left <= 5 else False
        near_right = True if job_info['img_shape'][1] - right <= 5 else False

        #如果贴边会把选择的背景图缩放到和人像图像一样大小然后替换原来的背景，和下面人景合成逻辑不同
        if near_top or near_left or near_right:
            if job_info['mode'] in ['photo','phone','coin']:
                resolution = 1152
            elif job_info['mode'] == 'puzzle':
                resolution = 1088
            elif job_info['mode'] == 'fridge':
                resolution = 800
            else:
                resolution = 1280
            input_img = resize_image(img, resolution=resolution)
            new_height, new_width = input_img.shape[:2]

            pred_alpha_ = np.expand_dims(pred_alpha, axis=2)
            alpha = cv2.resize(pred_alpha_, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            alpha = np.expand_dims(alpha, axis=2)

            mask = cv2.resize(pred_mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            ori_mask_final = np.expand_dims(mask, axis=2)

            back_img = cv2.resize(back_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            img_blend = input_img * alpha + back_img * (1 - alpha)
            img_blend_iter = img_blend.astype(np.uint8)

            num = 2
            for i in range(num):
                img_blend_iter = img_blend_iter * alpha + back_img * (1 - alpha)
                img_blend_iter = img_blend_iter.astype(np.uint8)
        else:
            #下面就是正常人景合成
            crop_alpha_ = np.expand_dims(crop_alpha, axis=2)
            alpha_ = cv2.resize(crop_alpha_, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            alpha_ = np.expand_dims(alpha_, axis=2)

            mask_ = cv2.resize(crop_mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            mask_ = np.expand_dims(mask_, axis=2)

            ori_alpha_final = np.zeros((h,w,1))
            ori_mask_final = np.zeros((h,w,1))
            source_img = np.zeros(back_img.shape)
            input_img = cv2.resize(crop_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            offset = int(w * 0.1)
            if local[0] == 0:
                if near_left:
                    offset_x = 0
                elif near_right:
                    offset_x = w - new_width
                elif new_width + offset < w:
                    offset_x = offset
                else:
                    offset_x = 0
            elif local[0] == 1:
                if near_left:
                    offset_x = 0
                elif near_right:
                    offset_x = w - new_width
                else: 
                    offset_x = (w - new_width) // 2
            else:
                if near_left:
                    offset_x = 0
                elif near_right:
                    offset_x = w - new_width
                elif new_width + offset < w:
                    offset_x = w - new_width - offset
                else:
                    offset_x = w - new_width
            
            crop_l = offset_x
            crop_r = crop_l + new_width

            offset_y = local[2]
            
            if new_height + offset_y < h:
                crop_t = h - offset_y - new_height
                crop_b = h - offset_y
            else:
                crop_t = h - new_height
                crop_b = h

            source_img[crop_t: crop_b, crop_l:crop_r, :] = input_img
            ori_alpha_final[crop_t: crop_b, crop_l:crop_r, :] = alpha_
            ori_mask_final[crop_t: crop_b, crop_l:crop_r, :] = mask_

            img_blend = source_img * ori_alpha_final + back_img * (1 - ori_alpha_final)
            img_blend_iter = img_blend.astype(np.uint8)

            job_info['blend'] = img_blend_iter

            #这里选择混合两次使边缘更自然
            num = 2
            for i in range(num):
                img_blend_iter = img_blend_iter * ori_alpha_final + back_img * (1 - ori_alpha_final)
                img_blend_iter = img_blend_iter.astype(np.uint8)

            job_info['blend1'] = img_blend_iter

        #下面做和谐化
        job_info['composite_img'] = img_blend_iter
        job_info['mask'] = ori_mask_final

        pred = self.image_harmony.run(job_info)
        job_info['harmony_img'] = pred

        if near_top or near_left or near_right:
            job_info['sd_width'] = new_width
            job_info['sd_height'] = new_height
        else:
            job_info['sd_width'] = w
            job_info['sd_height'] = h

        job_info['fusion'] = True
        
        #风格化
        log_name = 'portrait with style, '
        self.run_depth_and_hed_subgraph(job_info, log_name)
        
        #人脸修复
        try:
            self.sd_interface.facial_restoration(job_info)
        except:
            print('face detector failed')

        h, w = pred.shape[:2]
        offset = 80

        logo = job_info['logo_image']
        logo_h, logo_w = logo.shape[:2]

        width = int(0.7*logo_w)
        height = int(0.7*logo_h)

        offset_x = w - width - offset
        offset_y = h - height - offset
        rect = (offset_x, offset_y, offset_x+width, offset_y+height)

        img = job_info['output_img']
        img_merge = merge_img(img, logo, rect)
        job_info['output_img'] = img_merge


    def get_time_analysis(self):
        # subgraph
        print('-' * 20)
        for key, value in self.subgraph_timer_dict.items():
            print(f'{key} average time passed: {value.average_time}')
        print('-' * 20)
        # module
        for key, value in self.module_timer_dict.items():
            print(f'{key} average time passed: {value.average_time}')
        print('-' * 20)
