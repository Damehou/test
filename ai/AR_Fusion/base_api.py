# -*- coding:utf-8 -*-
import os
import bios
from abc import ABC, abstractclassmethod


class BaseAPI(ABC):

    def __init__(self):
        root_path = os.path.join(os.path.split(os.path.realpath(__file__))[0].split('AR_Fusion')[0], 'AR_Fusion')
        config_path = os.path.join(root_path, 'config.yaml')
        self.config = bios.read(config_path)
        #self.config = bios.read("config.yaml")
        self.debug_level = 0

    @abstractclassmethod
    def run(self, data):
        """
        run module normally
        Args:
            data:
        Returns:
        """
        pass

    @staticmethod
    def log():
        print("logging")
