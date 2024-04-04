import os
from transformers import pipeline

class TextPreProcess:
    def __init__(self,device) -> None:
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        self.translator = pipeline("translation", model=os.path.join(root_path, "ckpts/sd_models/opus-mt-zh-en"),device = device)
        # self.text_generator = pipeline("text-generation", model='repositories/gpt2',device = device)
        print('opus-mt-zh-en init success')
    def process(self,prompt,use_txt_generator=False):
        prompt = self.translator(prompt)[0]['translation_text']
        # if use_txt_generator:
            # prompt = self.text_generator(prompt)[0]['generated_text'] 
        return prompt
