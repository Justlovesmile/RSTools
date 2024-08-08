# import os
# os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
# import cv2
# import gc
# import time
# import math
import types
# import torch
import numpy as np
from torch import nn
from typing import List, Sequence, Union
# from itertools import product
# from math import ceil
# from osgeo import gdal,osr

from rstools.utils.split import get_multiscale_patch, slide_window
from rstools.utils.tif import read_RSTif

# from yolov5_obb.models.common import DetectMultiBackend
# from yolov5_obb.utils.general import non_max_suppression_obb
# from yolov5_obb.utils.torch_utils import select_device
# from yolov5_obb.utils.rboxs_utils import poly2rbox, rbox2poly

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def inference_detector_by_patches(
        model: nn.Module,
        imgs: ImagesType,
        sizes: List[int],
        steps: List[int],
        ratios: List[float],
        prepare_patch_data: types.FunctionType,
        detect_patchs: types.FunctionType,
        merge_nms: types.FunctionType,
        bs: int = 1):
    """inference large images in patches with the detector
    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]): Either image files or
            loaded images.
        sizes (list[int]): The sizes of patches.
        steps (list[int]): The steps between two patches.
        ratios (list[float]): Image resizing ratios for multi-scale detecting.
        nms_cfg (dict): nms config.
        bs (int): Batch size, must greater than or equal to 1.
    Returns:
        list[np.ndarray]: Detection results.
    """
    assert bs >= 1, 'The batch size must greater than or equal to 1'
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    
    result_list = []
    for img in imgs:
        if isinstance(img, str):
            img = read_RSTif(img,return_info=False,norm=True,norm_mode=1)
        height, width = img.shape[:2]
        sizes, steps = get_multiscale_patch(sizes, steps, ratios)
        patches = slide_window(width, height, sizes, steps)

        results = []
        start = 0
        while True:
            # prepare patch data
            end = min(start + bs, len(patches))
            patch_datas = prepare_patch_data(img, patches[start:end])

            # forward the model
            results.extend(detect_patchs(model,patch_datas))

            if end >= len(patches):
                break
            start += bs

        # merge results by nms
        if results:
            result_list.append(
                merge_nms(
                    results,
                    patches[:, :2]
                ))
        else:
            result_list.append([])
    return result_list
    

# def inference(
#         imgs,
#         weights,
#         outdir,
#         log_path,
#         taskid,
#         args,
#         device='0',      # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         conf_thres=0.2,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         classes=None,    # filter by class: --class 0, or --class 0 2 3
#         max_det=1000,    # maximum detections per image
#         agnostic_nms=False,  # class-agnostic NMS
#         sizes=[1024],
#         steps=[974],
#         ratios=[1.0],
#         dnn=False):
#     device = select_device(device) 
#     model = DetectMultiBackend(weights, device=device, dnn=dnn)
#     def prepare_patch_data(img,patches):
#         datas = []
#         for patch in patches:
#             x1,y1,x2,y2 = patch
#             datas.append(img[y1:y2,x1:x2,:].transpose(2,0,1))
#         datas = torch.from_numpy(np.array(datas)).to(model.device)
#         datas = datas.float()
#         datas /= 255
#         return datas
    
#     def detect_patchs(model, patch_datas):
#         with torch.no_grad():
#             pred = model(patch_datas)
#         return pred.cpu()

#     def merge_nms(results, patches):
#         pred = torch.zeros((1,len(results)*results[0].shape[0], results[0].shape[1]), device=results[0].device)
#         for idx, (res, pat) in enumerate(zip(results, patches)):
#             res[..., :2] = res[..., :2] + torch.tensor(pat[:2]).to(results[0].device)
#             pred[...,idx*results[0].shape[0]:(idx+1)*results[0].shape[0],:] = res
#         pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
#         return pred
    
#     if isinstance(imgs, str):
#         imgpath = imgs
#         imgs,*info = read_RSTif(imgpath,return_info=True,norm=True,norm_mode=1)
    
#     predictions = inference_detector_by_patches(
#         model,imgs,sizes,steps,ratios,prepare_patch_data,
#         detect_patchs,merge_nms,bs=1)
        
#     for pred in predictions:
#         print(f"Detected {len(pred)} objects!")
#         print(pred)
#         for i, det in enumerate(pred):  # per image
#             pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
#             det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])
#             visual_result(imgs,'visual.jpg',det.cpu().tolist())

# def polygonToRotRectangle(bbox):
#     """
#     :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
#     :return: Rotated Rectangle in format [cx, cy, w, h, theta]
#     """
#     bbox = np.array(bbox,dtype=np.float32)
#     bbox = np.reshape(bbox,newshape=(2,4),order='F')
#     angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])

#     center = [[0],[0]]

#     for i in range(4):
#         center[0] += bbox[0,i]
#         center[1] += bbox[1,i]

#     center = np.array(center,dtype=np.float32)/4.0

#     R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

#     normalized = np.matmul(R.transpose(),bbox-center)

#     xmin = np.min(normalized[0,:])
#     xmax = np.max(normalized[0,:])
#     ymin = np.min(normalized[1,:])
#     ymax = np.max(normalized[1,:])

#     w = xmax - xmin + 1
#     h = ymax - ymin + 1

#     return [float(center[0]),float(center[1]),w,h,angle]

# def visual_result(img,vis_path,results,colors=(0,255,0),scale=0.5,thickness=2,show_score=False):
#     for res in results:
#         x1, y1, x2, y2, x3, y3, x4, y4, score, cls_id = res
#         c = colors[int(cls_id)] if type(colors)==dict else colors
#         cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),c,thickness=thickness)
#         cv2.line(img,(int(x2),int(y2)),(int(x3),int(y3)),c,thickness=thickness)
#         cv2.line(img,(int(x3),int(y3)),(int(x4),int(y4)),c,thickness=thickness)
#         cv2.line(img,(int(x4),int(y4)),(int(x1),int(y1)),c,thickness=thickness)
#         if show_score and int(cls_id) != -3:
#             cv2.putText(img,f"{score:.2f}",(int(x2),int(y2)),cv2.FONT_HERSHEY_SIMPLEX,0.3,c,thickness)
#     if scale != 1.0:
#         img = cv2.resize(img,None,fx=scale,fy=scale)
#     cv2.imwrite(vis_path,img)


if __name__ == "__main__":
    # inference("xxxx.tiff",
    #           "yolov5n_csl_dotav1_best.pt",
    #           './',
    #           './test.log',
    #           '100000',
    #           None)
    pass
