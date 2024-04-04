
__all__ = ['LessFaceException', 'NoPersonException', 'ImageSizeUnqualifiedException']

from Deployment.Exceptions import CustomException

class AlogrithmInvalidException(CustomException):
    """
    算法返回的不合法
    """
    MAJOR_CODE = 5


class LessFaceException(AlogrithmInvalidException):
    """
    用户图像人脸数量不够
    """
    MIRROR_CODE = 0


class NoPersonException(AlogrithmInvalidException):
    """
    用户图像无人
    """
    MIRROR_CODE = 1


class ImageSizeUnqualifiedException(AlogrithmInvalidException):
    """
    不合理的图片尺寸
    """
    MIRROR_CODE = 2
    
