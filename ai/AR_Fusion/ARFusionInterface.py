import sys

from Deployment.Services.ImageFetchService import image_fetch
from Utils.image_convert_utils import image_to_base64
sys.path.insert(0, './AR_Fusion')
import random
import math
import cv2
import os
import torch
import numpy as np
import json
from fastapi import APIRouter,Form
from loguru import logger
from Utils.server_utils import base_interface_result_decorator, DictInterfaceResult
from PIL import Image
import uuid
from datetime import datetime

def get_random_items(data, prefix, suffixes):
    results = {}
    for suffix in suffixes:
        # 筛选出以指定前缀开头和后缀结尾的键
        filtered_keys = [key for key in data if key.startswith(prefix) and key.endswith(suffix + ".png")]
        # 如果有符合条件的键，随机选择一个
        if filtered_keys:
            selected_key = random.choice(filtered_keys)
            results[selected_key] = data[selected_key]
    return results

def concat_images_horizontally(*images):
    """
    Concatenate n OpenCV images horizontally without resizing original images.

    :param images: A variable number of OpenCV images.
    :return: A single image with all the input images concatenated side by side.
    """
    # 检查是否有图像，如果没有，则返回None
    if not images:
        return None

    # 获取所有图像的高度和宽度
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]

    # 找到最大的高度和宽度
    max_height = max(heights)
    max_width = max(widths)

    # 创建具有最大尺寸的背景图像
    backgrounds = [np.zeros((max_height, max_width, 3), dtype=np.uint8) for _ in images]

    # 将每个图像放置在背景中央
    for bg, img in zip(backgrounds, images):
        y_offset = (max_height - img.shape[0]) // 2
        x_offset = (max_width - img.shape[1]) // 2
        bg[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

    # 水平拼接图像
    concatenated_image = np.hstack(backgrounds)

    return concatenated_image


root_path = os.path.join(os.path.dirname(__file__),'..','..','AR_Fusion')
background_image_directory = os.path.join(root_path, 'background_images')
logo_image_directory = os.path.join(root_path, 'logo')
os.makedirs(f'{root_path}/images/input', exist_ok=True)
os.makedirs(f'{root_path}/images/output', exist_ok=True)

#-----------------------------------------------------------
from AR_Fusion.graph_pipelines.graph_defs import AIGCGraph

router = APIRouter(prefix='/aigc')

device = 0
aigc_graph = AIGCGraph()
aigc_graph.create_node(device)

@router.post('/generate')
@base_interface_result_decorator('generate image')
async def generate(
    requestId:str=Form(...),
    face_image_url:str=Form(...),
    # background_image_half_url:str=Form(...), 
    # background_image_half_annotation:str=Form(...),
    # background_image_whole_url:str=Form(...),  
    # background_image_whole_annotation:str=Form(...),
    mode_index:int=Form(...),
    regenerate_count:int=Form(...),
):
    with DictInterfaceResult() as to_return_result:
        job_info_origin = {
            'face_big': False,
            'face_rot': False,
            'face_deg': 0,
            'face_width': 0,
            "task_id": 0,
            "job_id": 0,
            "error_code": 0,
            "error_msg": "success",
            'img': None,
            'collection_img': None,
            "mode": ['photo', 'phone', 'puzzle', 'fridge', 'totebag', 'hoodie', 'coin'][mode_index-1],    # choice of 
            'merge_collection_flag': False, #True,
            'include_face': True,
            'pose_change_flag': False,
            'prompt': '4k, clear, distance', 
            'target_width':768,
            'target_height':1152
        }


        bg_dir = '/mnt/transfer/results_/westlake'
        totebag_bg_dir = '/mnt/transfer/results_/totebag'
        puzzle_dir = '/mnt/transfer/results_/westlake_puzzle'
        fridge_dir = '/mnt/transfer/results_/westlake_fridge'
        loc_path = '/mnt/transfer/results_/loc_dict.json'
        loc1_path = '/mnt/transfer/results_/loc_dict1.json'
        whole_dict = {}
        half_dict = {}
        with open(loc1_path, 'r') as json_file:
            loc_dict = json.load(json_file)

            for k, v in loc_dict.items():
                filename = k
                part = filename.split('_')[0]
                new_name = filename.split('_')[1]
                bg_path = os.path.join(bg_dir, new_name)
                if part == 'whole':
                    whole_dict[bg_path] = v
                else:
                    half_dict[bg_path] = v

        totebag_whole_dict = {}
        totebag_half_dict = {}
        with open(loc_path, 'r') as json_file:
            loc_dict_ = json.load(json_file)

            for k, v in loc_dict_.items():
                filename = k
                part = filename.split('_')[0]
                new_name = filename.split('_')[1]

                bg_path = os.path.join(totebag_bg_dir, new_name)

                if part == 'whole':
                    totebag_whole_dict[bg_path] = v
                else:
                    totebag_half_dict[bg_path] = v

        target_width = job_info_origin['target_width']
        target_height = job_info_origin['target_height']
        sd_width = math.ceil(target_width/64)*64
        sd_height = math.ceil(target_height/64)*64
        job_info_origin['sd_width'] = sd_width
        job_info_origin['sd_height'] = sd_height

        logo_path = f'{root_path}/test/single.png'
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        job_info_origin['logo_image'] = logo
        
        style_list = ['watercolor', 'real', 'cartoon', 'cyberpunk']
        job_info = job_info_origin.copy()


        image = image_fetch(face_image_url,'face image')
        cur_uuid = uuid.uuid1()
        cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        image.save(f'{root_path}/images/input/{cur_time}_{cur_uuid}.png')
        
        image = np.array(image)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        job_info['img'] = image
        job_info['img_shape'] = image.shape[:2]

        index_num = (regenerate_count-1) % 4

        if job_info['mode'] == 'photo':
            style = style_list[index_num]

            random_whole = get_random_items(loc_dict, "whole_", ["0", "1"])
            random_half = get_random_items(loc_dict, "half_", ["0", "1"])

            job_info['locs_half'] = []
            job_info['bgs_half'] = []
            for item in random_half.items():
                half_k, half_v = item
                half_name = half_k.split('-')[0]
                half_id = half_name.split('_')[1] + '.png'
                bg_half_path = os.path.join(bg_dir, half_id)
                back_img_half = cv2.imread(bg_half_path)
                job_info['locs_half'].append(half_v)
                job_info['bgs_half'].append(back_img_half)

            job_info['locs_whole'] = []
            job_info['bgs_whole'] = []
            for item in random_whole.items():
                whole_k, whole_v = item
                whole_name = whole_k.split('-')[0]
                whole_id = whole_name.split('_')[1] + '.png'
                bg_whole_path = os.path.join(bg_dir, whole_id)
                back_img_whole = cv2.imread(bg_whole_path)
                job_info['locs_whole'].append(whole_v)
                job_info['bgs_whole'].append(back_img_whole)
        elif job_info['mode'] == 'phone':
            style = style_list[index_num]

            random_whole = get_random_items(loc_dict, "whole_", ["0", "1"])
            random_half = get_random_items(loc_dict, "half_", ["0", "1"])

            job_info['locs_half'] = []
            job_info['bgs_half'] = []
            for item in random_half.items():
                half_k, half_v = item
                half_name = half_k.split('-')[0]
                half_id = half_name.split('_')[1] + '.png'
                bg_half_path = os.path.join(bg_dir, half_id)
                back_img_half = cv2.imread(bg_half_path)
                job_info['locs_half'].append(half_v)
                job_info['bgs_half'].append(back_img_half)

            job_info['locs_whole'] = []
            job_info['bgs_whole'] = []
            for item in random_whole.items():
                whole_k, whole_v = item
                whole_name = whole_k.split('-')[0]
                whole_id = whole_name.split('_')[1] + '.png'
                bg_whole_path = os.path.join(bg_dir, whole_id)
                back_img_whole = cv2.imread(bg_whole_path)
                job_info['locs_whole'].append(whole_v)
                job_info['bgs_whole'].append(back_img_whole)
        elif job_info['mode'] == 'puzzle':
            style = style_list[index_num]

            random_whole = get_random_items(loc_dict, "whole_", ["0", "1"])
            random_half = get_random_items(loc_dict, "half_", ["0", "1"])

            job_info['locs_half'] = []
            job_info['bgs_half'] = []
            for item in random_half.items():
                half_k, half_v = item
                half_name = half_k.split('-')[0]
                half_id = half_name.split('_')[1] + '.png'
                bg_half_path = os.path.join(puzzle_dir, half_id)
                back_img_half = cv2.imread(bg_half_path)
                job_info['locs_half'].append(half_v)
                job_info['bgs_half'].append(back_img_half)

            job_info['locs_whole'] = []
            job_info['bgs_whole'] = []
            for item in random_whole.items():
                whole_k, whole_v = item
                whole_name = whole_k.split('-')[0]
                whole_id = whole_name.split('_')[1] + '.png'
                bg_whole_path = os.path.join(puzzle_dir, whole_id)
                back_img_whole = cv2.imread(bg_whole_path)
                job_info['locs_whole'].append(whole_v)
                job_info['bgs_whole'].append(back_img_whole)

        elif job_info['mode'] == 'fridge':
            style = style_list[index_num]

            random_whole = get_random_items(loc_dict, "whole_", ["0", "1"])
            random_half = get_random_items(loc_dict, "half_", ["0", "1"])

            job_info['locs_half'] = []
            job_info['bgs_half'] = []
            for item in random_half.items():
                half_k, half_v = item
                half_name = half_k.split('-')[0]
                half_id = half_name.split('_')[1] + '.png'
                bg_half_path = os.path.join(fridge_dir, half_id)
                back_img_half = cv2.imread(bg_half_path)
                job_info['locs_half'].append(half_v)
                job_info['bgs_half'].append(back_img_half)

            job_info['locs_whole'] = []
            job_info['bgs_whole'] = []
            for item in random_whole.items():
                whole_k, whole_v = item
                whole_name = whole_k.split('-')[0]
                whole_id = whole_name.split('_')[1] + '.png'
                bg_whole_path = os.path.join(fridge_dir, whole_id)
                back_img_whole = cv2.imread(bg_whole_path)
                job_info['locs_whole'].append(whole_v)
                job_info['bgs_whole'].append(back_img_whole)

        elif job_info['mode'] == 'totebag':
            style = style_list[index_num]

            job_info['locs_half'] = []
            job_info['bgs_half'] = []

            random_item = random.choice(list(totebag_half_dict.items()))
            bg_half_path, half_v = random_item
            back_img_half = cv2.imread(bg_half_path)
            job_info['locs_half'].append(half_v)
            job_info['bgs_half'].append(back_img_half)

            job_info['locs_whole'] = []
            job_info['bgs_whole'] = []

            random_item = random.choice(list(totebag_whole_dict.items()))
            bg_whole_path, whole_v = random_item
            back_img_whole = cv2.imread(bg_whole_path)
            job_info['locs_whole'].append(whole_v)
            job_info['bgs_whole'].append(back_img_whole)
        
        elif job_info['mode'] == 'coin':
            # style = random.choice(style_list)
            style = style_list[0]

            random_whole = get_random_items(loc_dict, "whole_", ["0", "1"])
            random_half = get_random_items(loc_dict, "half_", ["0", "1"])

            job_info['locs_half'] = []
            job_info['bgs_half'] = []
            for item in random_half.items():
                half_k, half_v = item
                half_name = half_k.split('-')[0]
                half_id = half_name.split('_')[1] + '.png'
                bg_half_path = os.path.join(bg_dir, half_id)
                back_img_half = cv2.imread(bg_half_path)
                job_info['locs_half'].append(half_v)
                job_info['bgs_half'].append(back_img_half)

            job_info['locs_whole'] = []
            job_info['bgs_whole'] = []
            for item in random_whole.items():
                whole_k, whole_v = item
                whole_name = whole_k.split('-')[0]
                whole_id = whole_name.split('_')[1] + '.png'
                bg_whole_path = os.path.join(bg_dir, whole_id)
                back_img_whole = cv2.imread(bg_whole_path)
                job_info['locs_whole'].append(whole_v)
                job_info['bgs_whole'].append(back_img_whole)

        elif job_info['mode'] == 'hoodie':
            style = style_list[index_num]
  
        job_info['style_mode'] = style
        with torch.no_grad():
            aigc_graph.run_style_graph(job_info)
        
        # TODO:
        concat_img = job_info['output_img']
        cv2.imwrite(f'{root_path}/images/output/{cur_time}_{cur_uuid}.png', concat_img)
        
        # for i, img in enumerate(job_info['output_img_list']):
        image_base64_list = []
        image_base64_list.append(image_to_base64(Image.fromarray(job_info['output_img'][...,::-1],'RGB')))

        to_return_result.add_sub_result('images_base64', image_base64_list)
        to_return_result.add_sub_result('image_count', len(image_base64_list))
        return to_return_result
