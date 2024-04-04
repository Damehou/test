import torch
from PIL import Image
import numpy as np
import facer
import math
import cv2
from skimage import exposure
from skimage import transform as trans
import time


def topk_bbox(bboxes, n):
    areas = []
    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        xmin, ymin, xmax, ymax = bbox_int
        area = max(0, xmax - xmin) * max(0, ymax - ymin)
        areas.append(area)

    topk_indexes = np.argsort(areas)[::-1][:n]

    return topk_indexes

def concat_images_horizontally(images):
    # Get dimensions of all images
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with the calculated width and the maximum height of the images
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side on the new image
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return new_image


def bchw2hwc(images: torch.Tensor, nrows=None, border: int = 2,
             background_value: float = 0) -> torch.Tensor:
    """ make a grid image from an image batch.

    Args:
        images (torch.Tensor): input image batch.
        nrows: rows of grid.
        border: border size in pixel.
        background_value: color value of background.
    """
    assert images.ndim == 4  # n x c x h x w
    images = images.permute(0, 2, 3, 1)  # n x h x w x c
    n, h, w, c = images.shape
    if nrows is None:
        nrows = max(int(math.sqrt(n)), 1)
    ncols = (n + nrows - 1) // nrows
    result = torch.full([(h + border) * nrows - border,
                         (w + border) * ncols - border, c], background_value,
                        device=images.device,
                        dtype=images.dtype)

    for i, single_image in enumerate(images):
        row = i // ncols
        col = i % ncols
        yy = (h + border) * row
        xx = (w + border) * col
        result[yy:(yy + h), xx:(xx + w), :] = single_image
    return result

def crop_with_ldmk(img, ldmk, size=256):
    std_ldmk = np.array([[193, 240], [319, 240],
                         [257, 314], [201, 371],
                         [313, 371]], dtype=np.float32) / 512 * size
    tform = trans.SimilarityTransform()
    tform.estimate(ldmk, std_ldmk)
    M = tform.params[0:2, :]
    cropped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)
    invert_param = cv2.invertAffineTransform(M)
    return cropped, invert_param

# def get_face_box_and_mask(pil_image, face_detector, face_parser, device, with_hair=False, with_bbox=False):
#     index_list = [1,2,3,4,5,6,7,8,9,10,11,12]
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#     h, w = image.shape[:2]
#     bboxes, landmarks = face_detector.run(image)

#     size = 512
#     std_ldmk = np.array([[193, 240], [319, 240],
#                          [257, 314], [201, 371],
#                          [313, 371]], dtype=np.float32) / 512 * size
#     tform = trans.SimilarityTransform()

#     if with_bbox:
#         bbox = None
#         if bboxes.shape[0] > 0:
#             bbox = bboxes[0].astype(np.int32)
#         return bbox
#     else:
#         vis_img = np.zeros(image.shape)
#         for i in range(bboxes.shape[0]):
#             bbox = bboxes[i].astype(np.int32)
#             landmark = landmarks[i].astype(np.int32)
#             tform.estimate(landmark, std_ldmk)
#             M = tform.params[0:2, :]
#             inv_M = cv2.invertAffineTransform(M)

#             # import pdb;pdb.set_trace()
#             align_face = cv2.warpAffine(image, M, (size, size), borderValue=0.0)
#             face_mask = face_parser.run(align_face)
#             face_mask = face_mask.astype(np.uint8)
#             mask = cv2.warpAffine(face_mask, inv_M, (w, h), flags=cv2.INTER_NEAREST)

#             if not with_hair:
#                 mask[mask==13] = 0
#             else:
#                 mask[mask==13] = 1
            
#             for i in index_list:
#                 vis_img[np.where(mask==i)] = 255

#         vis_img = Image.fromarray(vis_img.astype(np.uint8))

#         return vis_img

def get_face_box_and_mask1(pil_image, bbox, landmark, face_parser, face_num):
    index_list = [1,2,3,4,5,6,7,8,9,10,11,12]
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]

    size = 512
    std_ldmk = np.array([[193, 240], [319, 240],
                         [257, 314], [201, 371],
                         [313, 371]], dtype=np.float32) / 512 * size
    tform = trans.SimilarityTransform()

    vis_img = np.zeros(image.shape)
    bbox = bbox.astype(np.int32)
    landmark = landmark.astype(np.int32)
    tform.estimate(landmark, std_ldmk)
    M = tform.params[0:2, :]
    inv_M = cv2.invertAffineTransform(M)

    align_face = cv2.warpAffine(image, M, (size, size), borderValue=0.0)
    face_mask = face_parser.run(align_face)
    face_mask = face_mask.astype(np.uint8)
    mask = cv2.warpAffine(face_mask, inv_M, (w, h), flags=cv2.INTER_NEAREST)
    
    index_mask = np.isin(mask, index_list)
    vis_img[index_mask] = 255

    vis_img = Image.fromarray(vis_img.astype(np.uint8))

    return vis_img, bbox

def get_face_box_and_mask(pil_image, bbox, landmark, face_parser, face_num):
    index_list = [1,2,3,4,5,6,7,8,9,10,11,12]
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]

    size = 512
    std_ldmk = np.array([[193, 240], [319, 240],
                         [257, 314], [201, 371],
                         [313, 371]], dtype=np.float32) / 512 * size
    tform = trans.SimilarityTransform()

    # vis_img = np.zeros(image.shape)
    bbox = bbox.astype(np.int32)
    landmark = landmark.astype(np.int32)
    tform.estimate(landmark, std_ldmk)
    M = tform.params[0:2, :]
    inv_M = cv2.invertAffineTransform(M)

    align_face = cv2.warpAffine(image, M, (size, size), borderValue=0.0)
    
    face_mask = face_parser.run(align_face)
    face_mask = face_mask.astype(np.uint8)
    mask = cv2.warpAffine(face_mask, inv_M, (w, h), flags=cv2.INTER_NEAREST)

    vis_img = np.zeros((h, w), dtype=np.uint8)
    index_mask = np.isin(mask, index_list)
    vis_img[index_mask] = 255

    if face_num == 1:
        oneface_mask = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR))
    else:
        jaw_points = landmark
        ellipse_mask = np.zeros_like(mask)
        (cx, cy), (width, height), angle = cv2.fitEllipse(jaw_points)
        center = (int(cx), int(cy))
        axes = (int(width / 2), int(height / 2))  # 注意，这里是半径，不是直径
        cv2.ellipse(ellipse_mask, center, axes, angle, 0, 360, 255, -1)  # 画实心椭圆

        # 计算每个连通分量与椭圆掩膜的交集面积，并保留交集最大的分量
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vis_img)
        max_intersect_area = 0
        max_label = 0
        for i in range(1, num_labels):  # 跳过背景
            intersect_mask = np.logical_and(labels == i, ellipse_mask)
            intersect_area = np.sum(intersect_mask)
            if intersect_area > max_intersect_area:
                max_intersect_area = intersect_area
                max_label = i
        
        # 创建一个新掩膜，仅包含面积最大的交集
        oneface_mask = np.where(labels == max_label, 255, 0).astype(np.uint8)
        oneface_mask = Image.fromarray(cv2.cvtColor(oneface_mask, cv2.COLOR_GRAY2BGR))

    return oneface_mask, bbox

# def get_face_box_and_mask(pil_image, face_detector, face_parser, device, with_hair=False, with_bbox=True):
#     index_list = [1,2,3,4,5,6,7,8,9,10,11,12]
#     image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#     h, w = image.shape[:2]
#     bboxes, landmarks = face_detector.run(image)

#     size = 512
#     std_ldmk = np.array([[193, 240], [319, 240],
#                          [257, 314], [201, 371],
#                          [313, 371]], dtype=np.float32) / 512 * size
#     tform = trans.SimilarityTransform()

#     vis_img = np.zeros(image.shape)
#     for i in range(bboxes.shape[0]):
#         bbox = bboxes[i].astype(np.int32)
#         landmark = landmarks[i].astype(np.int32)
#         tform.estimate(landmark, std_ldmk)
#         M = tform.params[0:2, :]
#         inv_M = cv2.invertAffineTransform(M)

#         align_face = cv2.warpAffine(image, M, (size, size), borderValue=0.0)
#         face_mask = face_parser.run(align_face)
#         face_mask = face_mask.astype(np.uint8)
#         mask = cv2.warpAffine(face_mask, inv_M, (w, h), flags=cv2.INTER_NEAREST)

#         if not with_hair:
#             mask[mask==13] = 0
#         else:
#             mask[mask==13] = 1
        
#         for i in index_list:
#             vis_img[np.where(mask==i)] = 255

#     vis_img = Image.fromarray(vis_img.astype(np.uint8))
#     # import pdb;pdb.set_trace()

#     bbox = None
#     if bboxes.shape[0] > 0:
#         bbox = bboxes[0].astype(np.int32)
#     return vis_img, bbox


def get_face_mask(pil_image, face_detector, face_parser, device, with_hair=True, with_bbox=False):
    
    np_image = np.array(pil_image.convert('RGB'))
    image = facer.hwc2bchw(torch.from_numpy(np_image).to(device=device)) # image: 1 x 3 x h x w

    # import ipdb; ipdb.set_trace()
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_parser(image, faces)

    #import ipdb; ipdb.set_trace()

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    #n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()
    
    if not with_hair:
        vis_seg_probs[vis_seg_probs==10] = 0.

    vis_seg_probs[vis_seg_probs > 0] = 255.
    vis_img = vis_seg_probs.sum(0, keepdim=True).float()
    
    vis_img = vis_img.unsqueeze(1)
    vis_img = bchw2hwc(vis_img)
    if vis_img.dtype != torch.uint8:
        vis_img = vis_img.to(torch.uint8)
    if vis_img.size(2) == 1:
        vis_img = vis_img.repeat(1, 1, 3)

    vis_img = Image.fromarray(vis_img.cpu().numpy())
    
    if with_bbox:
        bbox = None
        if len(faces['rects']) > 0:
            bbox = np.array(faces['rects'][0].cpu(), dtype=int)
        return vis_img, bbox
    else:
        return vis_img


def canny_process(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def crop_pil_image_with_bbox(image, bbox, crop_length=None):
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    length = max(x2 - x1, y2 - y1) if crop_length is None else crop_length
    half_length = length / 2

    new_x1 = np.clip(center[0] - half_length, 0, image.width)
    new_y1 = np.clip(center[1] - half_length, 0, image.height)
    new_x2 = np.clip(center[0] + half_length, 0, image.width)
    new_y2 = np.clip(center[1] + half_length, 0, image.height)

    return image.crop((new_x1, new_y1, new_x2, new_y2)), (new_x1, new_y1, new_x2, new_y2)


def paste_back(orig_image, cropped_image, cropped_bbox):
    x1, y1, x2, y2 = map(int, cropped_bbox)
    orig_image.paste(cropped_image, (x1, y1, x2, y2))

    return orig_image


def combine_images_with_mask(mask_image, image1, image2):
    """
    使用遮罩将图像1和图像2组合成新的图像。

    参数：
    mask_image (PIL.Image.Image): 遮罩图像,其中1表示使用图像1,0表示使用图像2
    image1 (PIL.Image.Image): 图像1
    image2 (PIL.Image.Image): 图像2

    返回：
    PIL.Image.Image: 组合后的新图像
    """
    # 将遮罩图像转换为NumPy数组
    mask_array = np.array(mask_image)

    # 将图像1和图像2转换为NumPy数组
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    index = np.where(mask_array == 255)
    try:
        top, bottom = index[0].min(), index[0].max()
        left, right = index[1].min(), index[1].max()

        width = right - left
        height = bottom - top

        ksize = max(int(min(width, height) * 0.075) | 1, 13)
    except ValueError:
        ksize = 13

    result_array = np.where(mask_array == 255, image1_array, image2_array)
    
    mask = cv2.GaussianBlur(mask_array.astype(np.float32), (ksize, ksize), 0)
    result_array = image1_array * mask/255. + image2_array * (1-mask/255.)

    # 创建包含组合图像的PIL图像
    result_image = Image.fromarray(result_array.astype('uint8'))

    return result_image


def color_transfer(src_img_pil, ref_img_pil, src_mask_pil, ref_mask_pil):
    
    # Convert PIL images to OpenCV format
    src = cv2.cvtColor(np.array(src_img_pil), cv2.COLOR_RGB2BGR)
    ref = cv2.cvtColor(np.array(ref_img_pil), cv2.COLOR_RGB2BGR)
    src_mask = cv2.cvtColor(np.array(src_mask_pil), cv2.COLOR_RGB2BGR)
    ref_mask = cv2.cvtColor(np.array(ref_mask_pil), cv2.COLOR_RGB2BGR)
    
    src_mask_bool = src_mask[:,:,0] > 128
    src_masked = src[src_mask_bool].reshape((1, -1, 3))

    ref_mask_bool = ref_mask[:,:,0] > 128
    ref_masked = ref[ref_mask_bool].reshape((1, -1, 3))

    matched_masked = exposure.match_histograms(src_masked, ref_masked, channel_axis=-1).reshape((-1, 3))

    matched = src.copy()
    matched[src_mask_bool] = matched_masked

    # Convert the OpenCV image back to PIL format
    matched_img_pil = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))

    return matched_img_pil

def color_transfer1(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.
    
    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image 

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc-s_mean)*(t_std/s_std))+t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst


def improved_color_transfer(src_img_pil, ref_img_pil, src_mask_pil, ref_mask_pil, blend_factor=0.5, sigma=1):
    # Convert PIL images to OpenCV format
    src = cv2.cvtColor(np.array(src_img_pil), cv2.COLOR_RGB2BGR)
    ref = cv2.cvtColor(np.array(ref_img_pil), cv2.COLOR_RGB2BGR)
    src_mask = cv2.cvtColor(np.array(src_mask_pil), cv2.COLOR_RGB2BGR)
    ref_mask = cv2.cvtColor(np.array(ref_mask_pil), cv2.COLOR_RGB2BGR)

    src_mask_bool = src_mask[:,:,0] > 128
    src_masked = src[src_mask_bool].reshape((1, -1, 3))

    ref_mask_bool = ref_mask[:,:,0] > 128
    ref_masked = ref[ref_mask_bool].reshape((1, -1, 3))

    # Perform histogram matching
    matched_masked = exposure.match_histograms(src_masked, ref_masked, channel_axis=-1).reshape((-1, 3))

    # Blend the matched image and the original image
    blended_masked = (1 - blend_factor) * src_masked + blend_factor * matched_masked
    blended_masked = np.clip(blended_masked, 0, 255)  # Ensure values are within valid range

    # Reconstruct the image with the blended colors
    matched = src.copy()
    matched[src_mask_bool] = blended_masked.reshape(-1, 3)

    # Generate edge mask
    edge_mask = cv2.Canny(src_mask, 100, 200)
    dilated_edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Apply Gaussian blur only on the edges
    blurred_matched = cv2.GaussianBlur(matched, (0, 0), sigma)
    matched = np.where(dilated_edge_mask[:, :, np.newaxis] == 255, blurred_matched, matched)

    # Convert the OpenCV image back to PIL format
    matched_img_pil = Image.fromarray(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))

    return matched_img_pil

def adjust_gamma(image_pil, gamma=1.0):
    # 建立查找表，实现gamma校正
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted_image_cv2 = cv2.LUT(image_cv, table)
    adjusted_pil = Image.fromarray(cv2.cvtColor(adjusted_image_cv2, cv2.COLOR_BGR2RGB))
    # 应用gamma校正
    return adjusted_pil
