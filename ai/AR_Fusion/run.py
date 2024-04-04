import cv2
import os
import time
from graph_pipelines.graph_defs import AIGCGraph
import numpy as np
import math
import json
from PIL import Image
import random

def resize_image(img, resolution=1024):
    image = img.copy()
    H, W, C = image.shape
    long_side = max(H, W)
    short_side = min(H, W)
    if long_side>= resolution:
        H = float(H)
        W = float(W)
        k = float(resolution) / max(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    else:
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
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

if __name__ == '__main__':

    # loc_dict = {'half_1.png': [0.28, 352, 0, 0.72], 'half_10.png': [0.3, 388, 0, 0.78], 
    #             'half_11.png': [0.31, 394, 0, 0.78], 'half_12.png': [0.3, 388, 0, 0.78], 
    #             'half_13.png': [0.3, 388, 0, 0.78], 'half_14.png': [0.3, 388, 0, 0.78], 
    #             'half_15.png': [0.3, 388, 0, 0.78], 'half_16.png': [0.3, 387, 0, 0.78], 
    #             'half_17.png': [0.3, 388, 0, 0.78], 'half_18.png': [0.32, 407, 0, 0.68], 
    #             'half_19.png': [0.3, 388, 0, 0.78], 'half_2.png': [0.31, 396, 0, 0.72], 
    #             'half_20.png': [0.3, 388, 0, 0.78], 'half_21.png': [0.31, 401, 0, 0.69], 
    #             'half_22.png': [0.33, 419, 0, 0.69], 'half_23.png': [0.33, 423, 0, 0.69], 
    #             'half_24.png': [0.33, 423, 0, 0.69], 'half_25.png': [0.33, 419, 0, 0.69], 
    #             'half_26.png': [0.33, 419, 0, 0.69], 'half_27.png': [0.33, 420, 0, 0.69], 
    #             'whole_28.png': [0.36, 462, 0, 0.62], 'whole_29.png': [0.38, 484, 0, 0.62], 
    #             'half_3.png': [0.31, 391, 0, 0.72], 'whole_30.png': [0.39, 504, 0, 0.63], 
    #             'half_31.png': [0.33, 419, 0, 0.69], 'half_32.png': [0.33, 419, 0, 0.69], 
    #             'half_33.png': [0.33, 419, 0, 0.69], 'half_34.png': [0.33, 419, 0, 0.69], 
    #             'half_35.png': [0.33, 420, 0, 0.69], 'half_36.png': [0.33, 420, 0, 0.69], 
    #             'half_37.png': [0.33, 419, 0, 0.69], 'half_38.png': [0.33, 420, 0, 0.69], 
    #             'half_39.png': [0.37, 471, 0, 0.62], 'half_4.png': [0.31, 392, 0, 0.72], 
    #             'half_40.png': [0.3, 386, 0, 0.69], 'whole_41.png': [0.37, 468, 0, 0.62], 
    #             'half_42.png': [0.3, 384, 0, 0.7], 'half_43.png': [0.3, 387, 0, 0.7], 
    #             'half_44.png': [0.3, 384, 0, 0.69], 'half_45.png': [0.35, 449, 0, 0.62], 
    #             'half_46.png': [0.34, 432, 0, 0.59], 'half_47.png': [0.32, 415, 0, 0.64], 
    #             'half_48.png': [0.3, 384, 0, 0.69], 'half_49.png': [0.3, 383, 0, 0.69], 
    #             'half_5.png': [0.31, 392, 0, 0.72], 'half_50.png': [0.3, 383, 0, 0.69], 
    #             'half_51.png': [0.3, 383, 0, 0.69], 'half_52.png': [0.3, 383, 0, 0.69], 
    #             'half_53.png': [0.33, 424, 0, 0.62], 'half_54.png': [0.35, 452, 0, 0.63], 
    #             'half_55.png': [0.35, 453, 0, 0.63], 'half_56.png': [0.35, 453, 0, 0.63], 
    #             'half_57.png': [0.35, 452, 0, 0.63], 'whole_58.png': [0.38, 488, 0, 0.63], 
    #             'whole_59.png': [0.32, 413, 0, 0.57], 'half_6.png': [0.31, 391, 0, 0.72], 
    #             'whole_60.png': [0.37, 476, 0, 0.57], 'half_61.png': [0.34, 436, 0, 0.62], 
    #             'half_62.png': [0.35, 446, 0, 0.63], 'half_63.png': [0.35, 452, 0, 0.63], 
    #             'whole_64.png': [0.38, 483, 0, 0.74], 'whole_65.png': [0.44, 560, 0, 0.57], 
    #             'half_66.png': [0.35, 452, 0, 0.64], 'whole_67.png': [0.45, 571, 0, 0.69], 
    #             'whole_68.png': [0.42, 536, 0, 0.57], 'half_69.png': [0.42, 532, 0, 0.61], 
    #             'half_7.png': [0.31, 391, 0, 0.72], 'half_70.png': [0.35, 446, 0, 0.59], 
    #             'half_71.png': [0.39, 497, 0, 0.5], 'half_8.png': [0.31, 391, 0, 0.72], 
    #             'half_9.png': [0.3, 388, 0, 0.78]}

    # for k,v in loc_dict.items():
    #     v[0] = 1
    # with open('loc.json', 'w', encoding='utf-8') as f:
    #     json.dump(loc_dict, f, ensure_ascii=False)
    # import pdb;pdb.set_trace()

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

    # save_dir = '/mnt/transfer/results_/westlake_fridge'
    # os.makedirs(save_dir, exist_ok=True)
    # img_list = sorted(os.listdir(bg_dir))
    # for img_id in img_list:
    #     img_path = os.path.join(bg_dir, img_id)
    #     img = cv2.imread(img_path)
    #     h,w = img.shape[:2]
    #     if h > w:
    #         image = cv2.resize(img, (512, 800), interpolation=cv2.INTER_AREA)
    #     else:
    #         image = cv2.resize(img, (800, 512), interpolation=cv2.INTER_AREA)
        
    #     save_path = os.path.join(save_dir, img_id)
    #     cv2.imwrite(save_path, image)

    # import pdb;pdb.set_trace()
    
    device = 0
    aigc_graph = AIGCGraph()
    aigc_graph.create_node(device)
    
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
        "mode": 'puzzle',    # choice of ['img2img', 'text2img']
        'merge_collection_flag': False, #True,
        'include_face': True,
        'pose_change_flag': False,
        'prompt': '4k, clear, distance', 
        'style_mode': 'watercolor', # choice of ['cyber', 'guoman', 'fantasy', 'golden', 'draw', 'jingtan','lineart','watercolor']
        'target_width':768,
        'target_height':1152
                      }

    target_width = job_info_origin['target_width']
    target_height = job_info_origin['target_height']
    sd_width = math.ceil(target_width/64)*64
    sd_height = math.ceil(target_height/64)*64
    job_info_origin['sd_width'] = sd_width
    job_info_origin['sd_height'] = sd_height

    logo_path = 'test/single.png'
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    job_info_origin['logo_image'] = logo

    # print(job_info_origin)
    style_list = ['watercolor', 'cartoon', 'real', 'cyberpunk']
    # style_list = ['cyber', 'cyberpunk', 'apoca_cyber', 'elem_cyber', 'art_cyber', 'portrait_cyber','arcana']
    mode_list = ['photo', 'phone', 'puzzle', 'fridge', 'totebag', 'hoodie', 'coin']

    save_dir = './results4'

    save_sd_dir = os.path.join(save_dir, 'stable_diffusion33')
    os.makedirs(save_sd_dir, exist_ok=True)

    save_swap_dir = os.path.join(save_dir, 'swap')
    os.makedirs(save_swap_dir, exist_ok=True)

    job_info = job_info_origin.copy()

    # import pdb;pdb.set_trace()
    job_info['mode'] = mode_list[0]
    job_info['infer_time'] = []

    face_dict = {}

    cnt = 0
    img_dir = '/mnt/transfer/data/SGHM/ar-fusion-test'
    # img_dir = '/mnt/transfer/images/input_rename1'
    # img_dir = '/mnt/transfer/data/SGHM/part_body'
    # img_dir = './test'
    img_list = sorted(os.listdir(img_dir))
    # img_list = ['other-2.png','other-12.png','other-13.png','other-14.png','other-15.png','other-16.png','other-17.png','other-22.png','other-20.png']
    for img_id in img_list:
        print(img_id)
        # if img_id != '5.png':continue
        # if img_id not in ['a4.jpg', 'a5.jpg']: continue
        if img_id not in ['0.png','1.png','2.png','3.png','4.png','5.png','6.png','half-13.png','half-14.png','half-15.png','half-16.png','half-25.png']: continue
        # if img_id not in ['1.png', 'half-9.png', 'half-10.png', 'half-16.png', 'half-19.png', 'half-24.png']:continue
        # if img_id not in ['a1.jpg','a2.jpg','a3.jpg','a4.jpg','a5.jpg','a6.jpg','a7.jpg','a8.jpg','a9.jpg','a10.jpg','a11.jpg','a12.jpg','a13.jpg','a14.jpg','a15.jpg','a16.jpg','3.png','6.png','half-17.png','half-19.png','other-14.png','other-15.png','other-16.png','right1.jpg','right2.jpg','right3.jpg']: continue
        # if img_id not in ['all1.jpg','all3.jpg','left1.jpg','left2.jpg','left4.jpg','leftright1.jpg','leftright2.jpg','leftright3.jpg','right1.jpg','right2.jpg','right3.jpg','3.png','4.png','5.png','6.png','7.png','half-0.png','half-1.png','half-2.png','half-3.png','half-17.png','half-18.png','half-19.png','half-20.png','half-21.png','half-22.png','half-23.png','half-24.png','other-0.png','other-4.png','other-5.png','other-6.png','other-14.png','other-15.png','other-16.png','other-17.png','other-18.png','whole-3.png','whole-4.png','whole-5.png','whole-6.png','whole-7.png','whole-8.png','whole-9.png','whole-10.png','whole-12.png']: continue
        # if img_id not in ['image_20.png','image_103.png','image_104.png','image_105.png','image_124.png','image_153.png','image_228.png','image_295.png','image_302.png','image_306.png','image_309.png','image_314.png','image_317.png','image_319.png','image_322.png','image_326.png','image_328.png','image_333.png','image_334.png','image_338.png','image_345.png','image_351.png','image_355.png','image_359.png','image_360.png','image_368.png','image_369.png','image_370.png','image_371.png','image_373.png','image_382.png','image_386.png','image_399.png','image_400.png','image_401.png','image_404.png','image_406.png','image_409.png','image_416.png','image_424.png','image_428.png','image_431.png','image_437.png','image_446.png','image_471.png']: continue
        img_path = os.path.join(img_dir, img_id)
    # img_path = 'template_imgs/x.png'
    # img_id = 'x.png'
        image = cv2.imread(img_path)
        job_info['img_path'] = img_path
        job_info['img'] = image
        job_info['img_shape'] = image.shape[:2]
        job_info['img_id'] = img_path

        if job_info['mode'] == 'photo':
            # style = random.choice(style_list)
            style = style_list[2]

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

            # half_id = 'half_20-1.png'
            # half_v = loc_dict[half_id]
            # bg_half_path = os.path.join(bg_dir, '20.png')
            # back_img_half = cv2.imread(bg_half_path)
            # job_info['locs_half'][1] = half_v
            # job_info['bgs_half'][1] = back_img_half


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

            # whole_id = 'whole_52-1.png'
            # whole_v = loc_dict[whole_id]
            # bg_whole_path = os.path.join(bg_dir, '52.png')
            # back_img_whole = cv2.imread(bg_whole_path)
            # job_info['locs_whole'][1] = whole_v
            # job_info['bgs_whole'][1] = back_img_whole


        elif job_info['mode'] == 'phone':
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
            
        elif job_info['mode'] == 'puzzle':
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
            # style = random.choice(style_list)
            style = style_list[2]

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

        job_info['style_mode'] = style
        aigc_graph.run_style_graph(job_info)
        # aigc_graph.run_normal_img2img_graph(job_info)

        # if 'face_len' in job_info.keys():
        #     face_dict[img_id] = job_info['face_len']

    #half-9.png,other-7.png,whole-1.png,whole-10.png,whole-12.png,whole-17.png,whole-18.png,whole-3.png,whole-5.png,whole-6.png,whole-8.png,whole-9.png

        img = job_info['output_img']
        new_img_id = style+'-'+img_id
        save_path = os.path.join(save_sd_dir, new_img_id)
        cv2.imwrite(save_path, img)

        # save_path = os.path.join(save_sd_dir, 'blend.png')
        # cv2.imwrite(save_path, job_info['blend'])

        # save_path = os.path.join(save_sd_dir, 'blend1.png')
        # cv2.imwrite(save_path, job_info['blend1'])

        # save_path = os.path.join(save_sd_dir, 'person.png')
        # cv2.imwrite(save_path, job_info['person'])

        # save_path = os.path.join(save_sd_dir, 'face_det.png')
        # cv2.imwrite(save_path, job_info['image'])

        # cnt += 1
        # if cnt == 5:break

        # new_img_id = 'cropped-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['cropped_img']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'refine_cropped-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['refine_cropped_img']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'ctr-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['ctr_img']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'facial-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['facial_img']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'mask-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['mask']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'face-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['facial']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'style-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['style']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'hed-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['hed']
        # cropped_img.save(cropped_img_path)

        # new_img_id = 'linear-'+img_id
        # cropped_img_path = os.path.join(save_sd_dir, new_img_id)
        # cropped_img = job_info['linear']
        # cropped_img.save(cropped_img_path)

        # img = job_info['blend1']
        # new_img_id = style+'-'+'blend1'+'-'+img_id
        # save_path = os.path.join(save_sd_dir, new_img_id)
        # cv2.imwrite(save_path, img)

        # img = job_info['blend2']
        # new_img_id = style+'-'+'blend2'+'-'+img_id
        # save_path = os.path.join(save_sd_dir, new_img_id)
        # cv2.imwrite(save_path, img)
        
        # if 'harmony_img_list' in job_info.keys():
        #     output_img_list = job_info['harmony_img_list']

        #     for i, img in enumerate(output_img_list):
        #         new_img_id = 'har-'+style+'_'+str(i)+'-'+img_id
        #         save_path = os.path.join(save_sd_dir, new_img_id)
        #         cv2.imwrite(save_path, img)
