import os
import sys
# import traceback
# from collections import namedtuple
# from pathlib import Path
# import re
import glob
import torch
import torch.hub
# from basicsr.utils.download_util import load_file_from_url
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# sys.path.insert(0,os.path.join(os.getcwd(), 'modules/sd_diffuser/blip'))
blip_image_eval_size = 384
blip_vqa_image_eval_size = 480
clip_model_name = 'ViT-L/14'

# Category = namedtuple("Category", ["name", "topn", "items"])

# re_topn = re.compile(r"\.top(\d+)\.")
def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    if ext_filter is None:
        ext_filter = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            if os.path.exists(place):
                for file in glob.iglob(place + '**/**', recursive=True):
                    full_path = file
                    if os.path.isdir(full_path):
                        continue
                    if os.path.islink(full_path) and not os.path.exists(full_path):
                        print(f"Skipping broken symlink: {full_path}")
                        continue
                    if ext_blacklist is not None and any([full_path.endswith(x) for x in ext_blacklist]):
                        continue
                    if len(ext_filter) != 0:
                        model_name, extension = os.path.splitext(file)
                        if extension not in ext_filter:
                            continue
                    if file not in output:
                        output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                dl = load_file_from_url(model_url, model_path, True, download_name)
                output.append(dl)
            else:
                output.append(model_url)

    except Exception:
        pass

    return output
class InterrogateModels:
    blip_model = None
    clip_model = None
    blip_vqa_model = None
    clip_preprocess = None
    dtype = None
    # running_on_cpu = None

    def __init__(self, ):
        # self.loaded_categories = None
        # self.skip_categories = []
        # self.content_dir = content_dir
        self.load()
    
    # def create_fake_fairscale(self):
    #     class FakeFairscale:
    #         def checkpoint_wrapper(self):
    #             pass
    #     sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    def load_blip_model(self):
        # self.create_fake_fairscale()
        import modules.sd_diffuser.blip.models.blip as blip
        # files =load_models(
        #     model_path=os.path.join('ckpts/sd_models', "BLIP"),
        #     model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
        #     ext_filter=[".pth"],
        #     download_name='model_base_caption_capfilt_large.pth',
        # )
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        file = os.path.join(root_path, 'ckpts/sd_models/BLIP/model_base_caption_capfilt_large.pth')
        blip_model = blip.blip_decoder(pretrained=file, image_size=blip_image_eval_size, vit='base', 
                                       med_config=os.path.join(root_path, 'modules/sd_diffuser/blip', "configs", "med_config.json"))
        blip_model.eval()
        print('blip_model load success')
        return blip_model

    def load_blip_vqa_model(self):
        # self.create_fake_fairscale()
        import modules.sd_diffuser.blip.models.blip_vqa as blip_vqa

        # files = load_models(
        #     model_path=os.path.join('ckpts/sd_models', "BLIP"),
        #     model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth',
        #     ext_filter=[".pth"],
        #     download_name='model_base_vqa_capfilt_large.pth',
        # )
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        file = os.path.join(root_path, 'ckpts/sd_models/BLIP/model_base_vqa_capfilt_large.pth')
        blip_vqa_model = blip_vqa.blip_vqa(pretrained=file, image_size=blip_vqa_image_eval_size, vit='base', 
                                           med_config=os.path.join(root_path, 'modules/sd_diffuser/blip', "configs", "med_config.json"))
        blip_vqa_model.eval()
        print('blip_vqa_model load success')
        return blip_vqa_model

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            
            self.blip_model = self.blip_model.half()

        self.blip_model = self.blip_model.to(torch.device("cuda", 0))
        self.dtype = next(self.blip_model.parameters()).dtype

        if self.blip_vqa_model is None:
            self.blip_vqa_model = self.load_blip_vqa_model()
            self.blip_vqa_model = self.blip_vqa_model.half()
        self.blip_vqa_model = self.blip_vqa_model.to(torch.device("cuda", 0))
        self.dtype = next(self.blip_vqa_model.parameters()).dtype


    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(torch.device("cuda", 0))

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=8, min_length=8, max_length=32)

        return caption[0]

    def interrogate_(self,pil_image):
        caption = self.generate_caption(pil_image)
        return caption


    def is_porn(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_vqa_image_eval_size, blip_vqa_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(torch.device("cuda", 0))
        # raw_image = Image.open(path).convert('RGB')
        with torch.no_grad():
            questions = ["is this an image from a porn?", "does this image contain tits and ass?",]
        return "yes" in [self.blip_vqa_model(gpu_image, q, train=False, inference='generate')[0] for q in questions]
    
    def gender_predict(self,pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_vqa_image_eval_size, blip_vqa_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(torch.device("cuda", 0))
        # raw_image = Image.open(path).convert('RGB')
        with torch.no_grad():
            # questions = ["is the person in this image male?", "is the person in this image female?",'how many persons in this image?']
            questions = ['is the person in this image male or female?','how many persons in this image?']
        return [self.blip_vqa_model(gpu_image, q, train=False, inference='generate')[0] for q in questions]
