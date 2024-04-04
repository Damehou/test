import sys
sys.path.insert(0, 'modules/sd_diffuser/ip_adapter/')

import os
import cv2
import glob
import torch
import random

import numpy as np
from abc import ABC
from PIL import Image
from safetensors.torch import load_file
from collections import defaultdict, namedtuple

from controlnet_aux import HEDdetector, LineartDetector, OpenposeDetector
from diffusers import ControlNetModel,EulerAncestralDiscreteScheduler
# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

# from modelscope.outputs import OutputKeys
# from modelscope.pipelines import pipeline

from base_api import BaseAPI
from modules.sd_diffuser.models.custome_pipeline_stable_diffusion_freeU import StableDiffusionCustomePipeline
from modules.sd_diffuser.models.utils import detectmap_proc
from modules.sd_diffuser.models.translator import TextPreProcess
from modules.sd_diffuser.models.interrogate import InterrogateModels
from modules.sd_diffuser.prompt_parser import parse_prompt_attention
from modules.sd_diffuser.ip_adapter import IPAdapterPlus
from modules.sd_diffuser.facial_restore.utils import (canny_process, improved_color_transfer,
                                                      crop_pil_image_with_bbox, combine_images_with_mask,
                                                      get_face_box_and_mask, topk_bbox, adjust_gamma)
from modules.codeformer.codeformer import CodeFormer


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []

PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])

def tokenize_line(line,pipeline):
    """
    this transforms a single prompt into a list of PromptChunk objects - as many as needed to
    represent the prompt.
    Returns the list and the total number of tokens in the prompt.
    """

    # import pdb
    # pdb.set_trace()
    parsed = parse_prompt_attention(line)
    comma_padding_backtrack =20
    chunk_length = 75
    comma_token = 267

    # pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    # tokenized = pipeline.text_encoder([text for text, _ in parsed])
    tokenized = [pipeline.tokenizer(text, return_tensors="pt", truncation=False).input_ids.tolist()[0][1:-1] for text,_ in parsed]
    chunks = []
    chunk = PromptChunk()
    token_count = 0
    last_comma = -1

    def find_embedding_at_position(tokens,offset):
        token = tokens[offset]
        ids_lookup = {}
        possible_matches = ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None
    
    def next_chunk(is_last=False):
        """puts current chunk into the list of results and produces the next one - empty;
        if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
        nonlocal token_count
        nonlocal last_comma
        nonlocal chunk
        id_end = 49407
        id_start = 49406
        if is_last:
            token_count += len(chunk.tokens)
        else:
            token_count += chunk_length

        to_add = chunk_length - len(chunk.tokens)
        if to_add > 0:
            chunk.tokens += [id_end] * to_add
            chunk.multipliers += [1.0] * to_add

        chunk.tokens = [id_start] + chunk.tokens + [id_end]
        chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

        last_comma = -1
        chunks.append(chunk)
        chunk = PromptChunk()

    for tokens, (text, weight) in zip(tokenized, parsed):
        if text == 'BREAK' and weight == -1:
            next_chunk()
            continue

        position = 0
        while position < len(tokens):
            token = tokens[position]

            if token == comma_token:
                last_comma = len(chunk.tokens)

            # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
            # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
            elif comma_padding_backtrack != 0 and len(chunk.tokens) == 75 and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                break_location = last_comma + 1

                reloc_tokens = chunk.tokens[break_location:]
                reloc_mults = chunk.multipliers[break_location:]

                chunk.tokens = chunk.tokens[:break_location]
                chunk.multipliers = chunk.multipliers[:break_location]

                next_chunk()
                chunk.tokens = reloc_tokens
                chunk.multipliers = reloc_mults

            if len(chunk.tokens) == chunk_length:
                next_chunk()

            embedding, embedding_length_in_tokens = find_embedding_at_position(tokens, position)
            if embedding is None:
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1
                continue

            emb_len = int(embedding.vec.shape[0])
            if len(chunk.tokens) + emb_len > chunk_length:
                next_chunk()

            chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

            chunk.tokens += [0] * emb_len
            chunk.multipliers += [weight] * emb_len
            position += embedding_length_in_tokens

    if len(chunk.tokens) > 0 or len(chunks) == 0:
        next_chunk(is_last=True)

    return chunks, token_count


class Stable_Diffusion(BaseAPI, ABC):
    def __init__(self, device=0):
        super().__init__()
        self.name = "stable_diffusion"
        self.device = torch.device("cuda", device) if torch.cuda.is_available() else "cpu"
        self.lora_dir = self.config['lora_dir']
        self.available_loras = dict()
        self.original_weights = dict()
        self.is_backup = False 
        self.cur_lora = ('', 0.0)
        self.repo_id = self.config['base_model_dir']
        self.dtype = torch.float16
        self.lora_map = {#'draw': 'draw_5_dm128', 
                         'watercolor': 'watercolor',
                         'cyberpunk': 'cyber1_d32_bs1_unet_and_te_w512_h512',
                         'real': 'realistic_portrait_realV51_d64_bs1-000010',
                        #  'cartoon': 'cartoon02_realisticV51_d128_bs1-000004',
                        #  'cartoon': 'cartoon02_realisticV51_d64_bs1-000018',
                         'cartoon': 'cartoon_portrait_v2',
                        #  'real': 'Asianbeauty2.0',
                        }

        self.pprompt_dicts = self.config['p_prompt']
        self.nprompt = self.config['n_prompt']
        self.lora_prompt_val =self.config['lora_prompt_val']
        self.img2img_params = self.config['img2img_params']
        self.pre_prompt = ''
        self.sensitive_words = self.config['sensitive_words'][0].split(',')
    
    def load(self, face_detector, face_parsing):
        self.preload_controlnet()
        self.pre_load_loras()
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        self.pipe = StableDiffusionCustomePipeline.from_pretrained(os.path.join(root_path, self.repo_id),
                                                                   torch_dtype=self.dtype,
                                                                   controlnet=None, variant='fp16',
                                                                   safety_checker=None)

        self.pipe.controlnet = MultiControlNetModel([self.hed_controlnet,
                                                     self.depth_controlnet])
        self.pipe.to(self.device)

        self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.ip_model = IPAdapterPlus(self.pipe, os.path.join(root_path, self.config['ip_image_encoder_ckpt']), 
                                      os.path.join(root_path, self.config['ip_adapter_ckpt']), self.device, num_tokens=16)
        
        self.translator = TextPreProcess(device=self.device)
        self.interrogate = InterrogateModels()
        self.set_scheduler()

        self.face_detector = face_detector
        self.face_parsing = face_parsing
        
        self.hed_preporcess = HEDdetector.from_pretrained(os.path.join(root_path, 'ckpts'))
        self.canny_preprocess = canny_process
        self.lineart_preprocess = LineartDetector.from_pretrained(os.path.join(root_path, 'ckpts'))
        # self.openpose_preprocess = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.codeformer = CodeFormer()
        self.codeformer.load()

    def set_scheduler(self,):
        # self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
    
    def pre_load_loras(self):
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        self.lora_dir = os.path.join(root_path, self.lora_dir)
        os.makedirs(self.lora_dir, exist_ok=True)

        candidates = glob.glob(os.path.join(self.lora_dir, '**/*.pt'), recursive=True) \
                        + glob.glob(os.path.join(self.lora_dir, '**/*.safetensors'), recursive=True) \
                        + glob.glob(os.path.join(self.lora_dir, '**/*.ckpt'), recursive=True)

        for filename in sorted(candidates, key=str.lower):
            if os.path.isdir(filename):
                continue

            name = os.path.splitext(os.path.basename(filename))[0]
            # load LoRA weight from .safetensors
            if 'cpu' != self.device:
                self.available_loras[name] = load_file(filename, device='cuda')
            else:
                self.available_loras[name] = load_file(filename, device='cpu')

    def back_up_weight(self):
        lora_name = self.lora_map[list(self.lora_map.keys())[0]]
        
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        state_dict = self.available_loras[lora_name]

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        index = 0
        # directly update weight in diffusers model
        for layer, elems in updates.items():
            index += 1

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.pipe.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.pipe.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)
            self.original_weights[index] = curr_layer.weight.data.clone().detach()
            

    def update_lora_weights(self,lora_name,multiplier):
        self.cur_lora = (lora_name,multiplier)
        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        if not lora_name:
            state_dict = self.available_loras[self.lora_map[list(self.lora_map.keys())[0]]]
        else:
            # lora_name = self.lora_map[list(self.lora_map.keys())[0]]
            state_dict = self.available_loras[lora_name]

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        index = 0
        # directly update weight in diffusers model
        for layer, elems in updates.items():
            index += 1

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.pipe.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.pipe.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)
            if not lora_name:
                curr_layer.weight.data = self.original_weights[index].clone().detach()
                continue
            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(self.dtype)
            weight_down = elems['lora_down.weight'].to(self.dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            curr_layer.weight.data = self.original_weights[index].clone().detach()

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
    
    def load_lora_weights(self, lora_name, multiplier):
        if not self.is_backup:
            self.back_up_weight()
            self.is_backup=True

        if lora_name:
            lora_name = self.lora_map[lora_name]
            if self.cur_lora == (lora_name, multiplier):
                return
            else:
                self.update_lora_weights(lora_name,multiplier)
        else:
            if self.cur_lora[0]:
                self.update_lora_weights(lora_name,multiplier)
            else:
                return

    def preload_controlnet(self, ):
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        self.depth_controlnet = ControlNetModel.from_pretrained(os.path.join(root_path, self.config['control_depth_dir']), 
                                                                torch_dtype=self.dtype).to(self.device)
        self.hed_controlnet = ControlNetModel.from_pretrained(os.path.join(root_path, self.config['control_hed_dir']), 
                                                              torch_dtype=self.dtype).to(self.device)
        self.canny_controlnet = ControlNetModel.from_pretrained(os.path.join(root_path, self.config['control_canny_dir']), 
                                                              torch_dtype=self.dtype).to(self.device)
        self.lineart_controlnet = ControlNetModel.from_pretrained(os.path.join(root_path, self.config['control_lineart_dir']), 
                                                                  torch_dtype=self.dtype).to(self.device)
        # self.openpose_controlnet = ControlNetModel.from_pretrained(os.path.join(root_path, self.config['control_openpose_dir']), 
        #                                                           torch_dtype=self.dtype).to(self.device)
        
    def prompt_preprocess(self,prompt,style,is_img2img=True):
        if not is_img2img:
            prompt = self.translator.process(prompt)
            print('[zh-en]:%s'%(prompt))
            word_s,word_e = 0,0
            new_prompt = ''
            word_nums = 0
            for i,ch in enumerate(prompt):
                if ch>='A' and ch<='z':
                    word_e = word_e+1
                    if i == len(prompt)-1:
                        word_nums = word_nums + 1
                        if prompt[word_s:word_e].lower() not in self.sensitive_words:
                            new_prompt = new_prompt+prompt[word_s:word_e]
                else:
                    word_nums = word_nums + 1
                    if prompt[word_s:word_e].lower() not in self.sensitive_words:
                        new_prompt = new_prompt+prompt[word_s:word_e]
                    new_prompt = new_prompt+prompt[word_e]
                    word_e = word_e + 1
                    word_s = word_e
            if word_nums <= 2:
                new_prompt = 'a photo of '+ new_prompt
            if new_prompt[-1] >='A' and new_prompt[-1]<='z':
                prompt = new_prompt
            else:
                prompt = new_prompt[:-1]
            print('[zh-en]:%s'%(prompt))
            self.pre_prompt = prompt
        if style == '':
            return prompt
        else:
            if style=='papercut':
                if is_img2img:
                    prompt = self.pprompt_dicts[style] + 'blue sky,white cloud,green grassland,an iconic scene from a Hayao Miyazaki film, inspired by Studio Ghibli style,'+prompt+','+'masterpiece,best quality,'
                else:
                    prompt = self.pprompt_dicts[style]+prompt+','+'masterpiece,best quality,'
            else:
                prompt = self.pprompt_dicts[style]+prompt
            # prompt = prompt+','+self.pprompt_dicts[style]
            return prompt
    
    def n_prompt_prepcoess(self,nprompt,prompt):
        if ('boy'in prompt or 'man' in prompt or 'boys' in prompt or 'mans' in prompt):
            nprompt = 'Feminized,female,girl,woman,girls,womans,'+nprompt
        return nprompt

    def process_token(self,pipeline,remade_batch_tokens,batch_multipliers):
        # tokens = torch.asarray(remade_batch_tokens).cuda()
        z = pipeline.text_encoder(torch.IntTensor(remade_batch_tokens).cuda().to(torch.int64))[0]
        batch_multipliers = torch.asarray(batch_multipliers).cuda()
        original_mean = z.mean()
        z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)
        return z 

    def get_pipeline_embed(self,pipeline,prompt,negative_prompt):
        embedd_chunks,_ = tokenize_line(prompt,pipeline)
        negative_embed_chunks,_ = tokenize_line(negative_prompt,pipeline)
        concat_embeds = []
        neg_embeds = []
        for i in range(len(embedd_chunks)):
            emd = self.process_token(pipeline,[embedd_chunks[i].tokens],[embedd_chunks[i].multipliers])
            concat_embeds.append(emd)
        for i in range(len(negative_embed_chunks)):
            emd = self.process_token(pipeline,[negative_embed_chunks[i].tokens],[negative_embed_chunks[i].multipliers])
            neg_embeds.append(emd)

        return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

    def get_ip_image_embed(self, init_image):
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(init_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, 1, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * 1, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, 1, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * 1, seq_len, -1)
        return image_prompt_embeds, uncond_image_prompt_embeds

    #sd+lora+controlnet，加载lora和controlnet做sd的图对图生成
    def img2img(self, prompt, gender_prompt, init_image, width, height, use_bclip=False, use_gender_just=True,
                style='', multiplier=0.0, steps=20, cfg_scale=7.5, sampler_index=0, denoising_strength=0.7, 
                use_depth=False, depth_image=None, depth_scale=0.0,
                use_hed=False, hed_image=None, hed_scale=0.0, 
                ip_scale=0.0, seed=None):
        self.style = style
        if self.style != 'real':
            self.load_lora_weights(style,multiplier)
        # self.load_lora_weights(style,multiplier)

        # generate image
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        init_image = Image.fromarray(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))
        init_image = init_image.resize((width, height))
        control_images_lst, control_scale_lst = [], []
        if use_hed:
            hed_image = detectmap_proc(hed_image, h=init_image.size[1], w=init_image.size[0])
            hed_image = torch.permute(hed_image, (1, 2, 0)).numpy()
            hed_image_uint8 = (hed_image * 255).astype('uint8')
            hed_image = Image.fromarray(hed_image_uint8)
            control_images_lst.append(hed_image)
            control_scale_lst.append(hed_scale)
        if use_depth:
            depth_image = detectmap_proc(depth_image, h=init_image.size[1], w=init_image.size[0])
            depth_image = torch.permute(depth_image, (1, 2, 0)).numpy()
            depth_image_uint8 = (depth_image * 255).astype('uint8')
            depth_image = Image.fromarray(depth_image_uint8)
            control_images_lst.append(depth_image)
            control_scale_lst.append(depth_scale)

        use_canny = True
        if use_canny:
            canny_image = self.canny_preprocess(init_image)
            control_images_lst.append(canny_image)
            control_scale_lst.append(0.6)

        if use_bclip:
            prompt = self.interrogate.interrogate_(init_image)
            self.pre_prompt = prompt
        # prompt preprocess
        if use_gender_just:
            prompt = 'asian, beautiful face,' + prompt
            # prompt = gender_prompt + prompt
        
        self.ip_model.pipe.controlnet = MultiControlNetModel([self.hed_controlnet,
                                                                self.depth_controlnet,
                                                                self.canny_controlnet])

        ## sensetive words delete
        prompt = self.prompt_preprocess(prompt, style)
        negative_prompt =  self.nprompt[0]
        print('prompt:%s\n'%prompt)

        prompt_embeds, negative_prompt_embeds = self.get_pipeline_embed(self.pipe, prompt, negative_prompt)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_ip_image_embed(init_image)
        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        self.ip_model.set_scale(ip_scale)
        images = self.ip_model.pipe(image=init_image,
                                    strength=denoising_strength,
                                    guidance_scale=cfg_scale,
                                    num_inference_steps=steps,
                                    generator=generator,
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    control_image=control_images_lst,
                                    controlnet_conditioning_scale=control_scale_lst,
                                    run_mode='ctrl_img2img').images
        return images

    def facial_restoration(self, job_info, seed=-1, steps=20, strength=0.55,
                           control_scale_lst=[0.9, 0.9]):
        ori_img = job_info['img_ori']
        ori_image_pil = Image.fromarray(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
        style_img = job_info['output_img']
        style_image_pil = Image.fromarray(cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB))
        final_style_img = style_image_pil.copy()

        topk_bboxes = job_info['topk_bbox']
        topk_landmarks = job_info['topk_landmarks']
        topk_genders = job_info['topk_genders']
        topk_ages = job_info['topk_ages']
        
        face_num = topk_bboxes.shape[0]

        if job_info['fusion']:
            before_style_image = job_info['composite_img']
        else:
            before_style_image = job_info['composed_img']
        style_bboxes, style_landmarks = self.face_detector.run(before_style_image)
        before_style_image_pil = Image.fromarray(cv2.cvtColor(before_style_image, cv2.COLOR_BGR2RGB))

        if face_num >= 2:
            topk_indexes = topk_bbox(style_bboxes, face_num)
            style_bboxes, style_landmarks = style_bboxes[topk_indexes], style_landmarks[topk_indexes]

        if style_bboxes.shape[0] < face_num:
            face_num = style_bboxes.shape[0]

        for i in range(face_num):
            bbox = topk_bboxes[i]
            landmark = topk_landmarks[i]
            gender = topk_genders[i]
            age = topk_ages[i]

            vis_img, bbox = get_face_box_and_mask(ori_image_pil, bbox, landmark, self.face_parsing, face_num)
            crop_length = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            cropped_head_image, _ = crop_pil_image_with_bbox(ori_image_pil, bbox, crop_length=crop_length*1.3)
            cropped_head_image = cropped_head_image.resize((256, 256))
            cropped_head_mask_image, _ = crop_pil_image_with_bbox(vis_img, bbox, crop_length=crop_length*1.3)
            cropped_head_mask_image = cropped_head_mask_image.resize((256, 256), Image.NEAREST)

            if gender == 1:
                if job_info['style_mode'] == 'cartoon':
                    if age >= 15:
                        prompt = 'best quality, cartoon_portrait, 1girl, solo, looking at viewer, beautiful brown eyes, ultra high res'
                    else:
                        prompt = 'best quality, cartoon_portrait, 1girl, solo, looking at viewer, brown eyes, ultra high res'
                else:
                    if age >= 15:
                        prompt = 'best quality, ultra high res, beautiful face, Detailed facial details, beautiful eyes'
                    else:
                        prompt = 'best quality, ultra high res, detailed face'
            else:
                if job_info['style_mode'] == 'cartoon':
                    prompt = 'best quality, cartoon_portrait, 1boy, solo, looking at viewer, beautiful brown eyes, ultra high res'
                elif job_info['style_mode'] == 'real':
                    prompt = 'best quality, ultra high res, handsome detailed face, (realistic), high detail, sharp focus, portrait of a man'
                elif job_info['style_mode'] == 'watercolor' or 'cyberpunk':
                    prompt = 'best quality, ultra high res, handsome detailed face, intricate, high detail, sharp focus'
            
            prompt = self.prompt_preprocess(prompt, self.style)
            if job_info['style_mode'] == 'cartoon':
                negative_prompt = 'FastNegativeV2,(bad-artist:1),(worst quality, low quality:1.4),(bad_prompt_version2:0.8),bad-hands-5,lowres,bad anatomy,bad hands,((text)),(watermark),error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,((username)),blurry,(extra limbs),bad-artist-anime,badhandv4,EasyNegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3,BadDream,(three hands:1.6),(three legs:1.2),(more than two hands:1.4),(more than two legs,:1.2),abel'
            else:
                negative_prompt = 'BadDream, (UnrealisticDream:1.3), FastNegativeV2, EasyNegative, paintings, sketches, ugly, 3d, (worst quality:1.3), (low quality:1.3), (normal quality:1.3), lowres, normal quality, (ugly:1.3), (morbid:1.2), deformed'
            preprocess_lst = [self.hed_preporcess, self.lineart_preprocess]
            controlnet = MultiControlNetModel([self.hed_controlnet, self.lineart_controlnet])

            bbox = style_bboxes[i]
            landmark = style_landmarks[i]
            vis_img, bbox = get_face_box_and_mask(style_image_pil, bbox, landmark, self.face_parsing, face_num)
            crop_length = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
            cropped_image, rec = crop_pil_image_with_bbox(style_image_pil, bbox, crop_length=crop_length*2)
            cropped_before_image, _ = crop_pil_image_with_bbox(before_style_image_pil, bbox, crop_length=crop_length*2)
            cropped_before_image = cropped_before_image.resize((512, 512), Image.LANCZOS)

            image = cv2.cvtColor(np.array(cropped_before_image), cv2.COLOR_RGB2GRAY)
            blur_measure = cv2.Laplacian(image, cv2.CV_64F).var()

            if blur_measure < 50:
                refine_cropped_before_array = self.codeformer.run(np.array(cropped_before_image))
            else:
                refine_cropped_before_array = np.array(cropped_before_image)
            refine_cropped_before_image = Image.fromarray(refine_cropped_before_array).resize((400, 400), Image.LANCZOS)
            
            x1, y1, x2, y2 = map(int, rec)
            x2 = x1 + cropped_image.size[0]
            y2 = y1 + cropped_image.size[1]
            cropped_image_400 = cropped_image.resize((400, 400))
            cropped_mask_image, _ = crop_pil_image_with_bbox(vis_img, bbox, crop_length=crop_length*2)
            cropped_mask_image = cropped_mask_image.resize((400, 400), Image.NEAREST)

            ctrl_images_lst = []
            for preprocess in preprocess_lst:
                ctrl_image = preprocess(cropped_image_400)
                ctrl_image = ctrl_image.resize(cropped_image_400.size)

                ctrl_before_image = preprocess(refine_cropped_before_image)
                ctrl_before_image = ctrl_before_image.resize(refine_cropped_before_image.size)

                # ctrl_image = (1 - np.array(cropped_mask_image)/255.) * np.array(ctrl_image)
                ctrl_image = (1 - np.array(cropped_mask_image)/255.) * np.array(ctrl_image) + \
                            np.array(cropped_mask_image)/255. * ctrl_before_image 
                ctrl_image = ctrl_image.astype(np.uint8)
                ctrl_image = Image.fromarray(ctrl_image)
                ctrl_images_lst.append(ctrl_image)

            cropped_new_head_image = cropped_head_image
            lora_weight = self.config['lora_face_restore'][job_info['style_mode']]
            if job_info['style_mode'] != 'real':
                self.load_lora_weights(job_info['style_mode'], lora_weight)
            prompt_embeds, negative_prompt_embeds = self.get_pipeline_embed(self.pipe, prompt, negative_prompt)
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_ip_image_embed(cropped_new_head_image)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

            if job_info['style_mode'] == 'cartoon':
                self.ip_model.set_scale(0.1)
            else:
                self.ip_model.set_scale(0.8)

            generator = torch.Generator(device=self.device).manual_seed(seed) if seed != -1 else None
            self.ip_model.pipe.controlnet = controlnet
            facial_image = self.ip_model.pipe(image=cropped_image_400,
                                            strength=strength,
                                            num_inference_steps=steps,
                                            generator=generator,
                                            prompt_embeds=prompt_embeds,
                                            negative_prompt_embeds=negative_prompt_embeds,
                                            control_image=ctrl_images_lst,
                                            controlnet_conditioning_scale=control_scale_lst,
                                            run_mode='ctrl_img2img').images[0]
            
            # color_image = improved_color_transfer(facial_image, cropped_image_400,
            #                                       cropped_mask_image, cropped_mask_image, 
            #                                       blend_factor=0.6)
            # resized_facial_image = color_image.resize(cropped_image.size)
            resized_facial_image = facial_image.resize(cropped_image.size)
            resized_cropped_mask_image = cropped_mask_image.resize(cropped_image.size)

            w, h = style_image_pil.size
            blank_mask_image = Image.new('RGB', (w, h), (0, 0, 0))
            blank_facial_image = Image.new('RGB', (w, h), (0, 0, 0))

            blank_mask_image.paste(resized_cropped_mask_image, (x1, y1))
            blank_facial_image.paste(resized_facial_image, (x1, y1))

            final_style_img = combine_images_with_mask(blank_mask_image, blank_facial_image, final_style_img)
        
        if job_info['style_mode'] == 'watercolor':
            gamma = 1.2
        elif job_info['style_mode'] == 'cyberpunk':
            gamma = 1.6
        else:
            gamma = 1
        adj_style_img = adjust_gamma(final_style_img.copy(), gamma=gamma)
        job_info['output_img'] = np.array(adj_style_img)[:, :, ::-1].copy()

    def run(self,):
        pass


if __name__ == "__main__":
    sd = Stable_Diffusion()
    sd.load()
    prompt = 'a dog'
    job_info = {}
    result = sd.txt2img(job_info,prompt,480,800,cfg_scale=7,steps=20,seed=1234)
    result_img = np.array(result)[:, :, ::-1]
    cv2.imwrite(prompt+'.jpg',result_img)
