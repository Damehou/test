
from time import sleep

import cv2
import numpy as np
from PIL import Image

from Utils.InferenceUtils.TritonHelper import TritonInferenceHelper


class DummyHandler:

    def __init__(self,
                 _model_name='fake_model', _model_version=1,
                 _triton_url='localhost', _triton_grpc_port=8001):
        self.inference_helper = TritonInferenceHelper(
            'dummy_model',
            _triton_url, _triton_grpc_port,
            _model_name, _model_version
        )
        self.inference_helper.add_input('INPUT__0', (3, 256, 256), '原始图像')
        self.inference_helper.add_output('OUTPUT__0', (3, 256, 256), '生成图像')

        assert _model_name.lower() in {'fake_model', }, 'not support model name'

    def warm_up(self, _warm_up_times=5):
        while True:
            try:
                self.inference_helper.check_ready()
                raw_test_image = Image.new('RGB', (256, 256))
                for i in range(_warm_up_times):
                    self(np.asarray(raw_test_image))
                break
            except Exception as e:
                sleep(1)
                print(e)
                continue

    def __call__(self, _image_np):
        if len(_image_np.shape) == 2:
            _image_np = cv2.cvtColor(_image_np, cv2.COLOR_GRAY2RGB)
        image_tensor = _image_np.astype(np.float32)
        image_tensor = np.transpose(image_tensor, (2, 0, 1))[None, ...]
        result = self.inference_helper.infer(False, False,
                                             data=image_tensor,
                                             )
        to_return_image = result['OUTPUT__0'][0]
        return to_return_image
