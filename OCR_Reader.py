import random
import cv2
import glob
import json
import os
from tqdm import tqdm
import time
from PIL import Image
import sys
import numpy as np
import pandas as pd

# # %% [markdown]
# # ## Load Yolov7 model

# # %% [code] {"execution":{"iopub.status.busy":"2023-10-14T21:10:16.536047Z","iopub.execute_input":"2023-10-14T21:10:16.536472Z","iopub.status.idle":"2023-10-14T21:10:16.542434Z","shell.execute_reply.started":"2023-10-14T21:10:16.536434Z","shell.execute_reply":"2023-10-14T21:10:16.541310Z"},"jupyter":{"outputs_hidden":false}}
# class Configs:
#     conf_thres=0.3
#     iou_thres=0.65
#     classes=[0,1,2,3,4,5,6,7]
#     agnostic_nms=False
#     augment=False
# optyolo = Configs

# # %% [code] {"papermill":{"duration":15.12419,"end_time":"2023-06-19T15:44:51.890204","exception":false,"start_time":"2023-06-19T15:44:36.766014","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-10-14T21:10:56.343314Z","iopub.execute_input":"2023-10-14T21:10:56.343671Z","iopub.status.idle":"2023-10-14T21:11:10.235631Z","shell.execute_reply.started":"2023-10-14T21:10:56.343641Z","shell.execute_reply":"2023-10-14T21:11:10.234690Z"},"jupyter":{"outputs_hidden":false}}
# sys.path.append("/kaggle/input/yolov7")
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_file, check_img_size, check_requirements, \
#     box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr

# device="cuda:0"
# # half = device != 'cpu'  # half precision only supported on CUDA
# yolo_model = attempt_load("/kaggle/input/yolov7-weights/yolov7_best.pt", map_location=device)  # load FP32 model
# stride = int(yolo_model.stride.max())  # model stride
# imgsz = check_img_size(640, s=stride)  # check img_size
# yolo_model.eval()

# # %% [code] {"papermill":{"duration":0.017879,"end_time":"2023-06-19T15:44:51.91809","exception":false,"start_time":"2023-06-19T15:44:51.900211","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-10-14T21:11:33.762118Z","iopub.execute_input":"2023-10-14T21:11:33.762681Z","iopub.status.idle":"2023-10-14T21:11:33.768322Z","shell.execute_reply.started":"2023-10-14T21:11:33.762647Z","shell.execute_reply":"2023-10-14T21:11:33.766811Z"},"jupyter":{"outputs_hidden":false}}
# sys.path.remove("/kaggle/input/yolov7")

# # %% [code] {"execution":{"iopub.status.busy":"2023-06-21T07:31:24.629502Z","iopub.execute_input":"2023-06-21T07:31:24.629867Z","iopub.status.idle":"2023-06-21T07:31:24.643043Z","shell.execute_reply.started":"2023-06-21T07:31:24.629835Z","shell.execute_reply":"2023-06-21T07:31:24.642002Z"},"jupyter":{"outputs_hidden":false}}
# import torch
# def detect_yolo(imgpth):
#     dataset = LoadImages(imgpth, img_size=imgsz, stride=stride)
#     old_img_w = old_img_h = imgsz
#     old_img_b = 1
#     for path, img, im0, vid_cap in dataset:
#         img = torch.from_numpy(img).to(device)
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#             pred = yolo_model(img.to("cuda:0"), augment=optyolo.augment)[0]
#         pred = non_max_suppression(pred, optyolo.conf_thres, optyolo.iou_thres, classes=optyolo.classes, agnostic=optyolo.agnostic_nms)
#         for i, det in enumerate(pred):  # detections per image
#             if len(det):
#             # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#                 det[:,2] = det[:,2] - det[:,0]
#                 det[:,3] = det[:,3] - det[:,1]
#                 det[:,0] = det[:,0] + det[:,2]/2
#                 det[:,1] = det[:,1] + det[:,3]/2

#         return pred[0].cpu()

# %% [markdown]
# ## load test images

# %% [code] {"papermill":{"duration":0.020137,"end_time":"2023-06-19T15:44:52.004594","exception":false,"start_time":"2023-06-19T15:44:51.984457","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-06-21T07:31:24.647242Z","iopub.execute_input":"2023-06-21T07:31:24.64753Z","iopub.status.idle":"2023-06-21T07:31:24.661826Z","shell.execute_reply.started":"2023-06-21T07:31:24.647505Z","shell.execute_reply":"2023-06-21T07:31:24.660827Z"},"jupyter":{"outputs_hidden":false}}
import glob
# all_imgs=glob.glob("/kaggle/input/test-1/Weixin Image_20231014204722.png")

# %% [markdown]
# ## Load EasyOCR model

# %% [code] {"papermill":{"duration":0.017099,"end_time":"2023-06-19T15:44:52.058067","exception":false,"start_time":"2023-06-19T15:44:52.040968","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-06-21T07:31:24.664848Z","iopub.execute_input":"2023-06-21T07:31:24.66511Z","iopub.status.idle":"2023-06-21T07:31:24.671121Z","shell.execute_reply.started":"2023-06-21T07:31:24.665087Z","shell.execute_reply":"2023-06-21T07:31:24.668904Z"},"jupyter":{"outputs_hidden":false}}
import sys
sys.path.append("/easyocr-clone")
sys.path.append("/easyocr-clone/trainer")
sys.path.append("/easyocr-clone/easyocr/model")
sys.path.append("/")

# %% [code] {"papermill":{"duration":0.065921,"end_time":"2023-06-19T15:44:52.133726","exception":false,"start_time":"2023-06-19T15:44:52.067805","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-06-21T07:31:24.673071Z","iopub.execute_input":"2023-06-21T07:31:24.673725Z","iopub.status.idle":"2023-06-21T07:31:24.755009Z","shell.execute_reply.started":"2023-06-21T07:31:24.673674Z","shell.execute_reply":"2023-06-21T07:31:24.754111Z"},"jupyter":{"outputs_hidden":false}}
from easyocr_clone.easyocr.model.model import Model
from easyocr_clone.trainer.utils import AttrDict
import yaml
import torch
from easyocr_clone.trainer.utils import CTCLabelConverter, AttnLabelConverter, Averager

# %% [code] {"papermill":{"duration":0.020551,"end_time":"2023-06-19T15:44:52.164327","exception":false,"start_time":"2023-06-19T15:44:52.143776","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-06-21T07:31:24.756758Z","iopub.execute_input":"2023-06-21T07:31:24.757024Z","iopub.status.idle":"2023-06-21T07:31:24.768523Z","shell.execute_reply.started":"2023-06-21T07:31:24.757001Z","shell.execute_reply":"2023-06-21T07:31:24.767497Z"},"jupyter":{"outputs_hidden":false}}
def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

# %% [code] {"papermill":{"duration":0.027551,"end_time":"2023-06-19T15:44:52.201677","exception":false,"start_time":"2023-06-19T15:44:52.174126","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-06-21T07:31:24.77191Z","iopub.execute_input":"2023-06-21T07:31:24.772171Z","iopub.status.idle":"2023-06-21T07:31:24.791944Z","shell.execute_reply.started":"2023-06-21T07:31:24.772148Z","shell.execute_reply":"2023-06-21T07:31:24.791105Z"},"jupyter":{"outputs_hidden":false}}
opt = get_config("./easyocr_finetuned_dataset/en_filtered_config.yaml")
device = torch.device('cpu')
if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-21T07:31:27.190262Z","iopub.execute_input":"2023-06-21T07:31:27.190965Z","iopub.status.idle":"2023-06-21T07:31:27.835248Z","shell.execute_reply.started":"2023-06-21T07:31:27.190929Z","shell.execute_reply":"2023-06-21T07:31:27.834332Z"},"jupyter":{"outputs_hidden":false}}
ocr_model = Model(input_channel = opt.input_channel, output_channel=opt.output_channel, hidden_size = opt.hidden_size, num_class = opt.num_class)
statedict = torch.load("./easyocr_finetuned_dataset/best_accuracy.pth",map_location=device)
newstatedict = {}
for k in statedict:
    newk = k[7:]
    newstatedict[newk] =statedict[k]
ocr_model.load_state_dict(newstatedict)
ocr_model.eval()
1

# %% [code] {"execution":{"iopub.status.busy":"2023-06-21T07:31:30.126497Z","iopub.execute_input":"2023-06-21T07:31:30.12713Z","iopub.status.idle":"2023-06-21T07:31:30.135833Z","shell.execute_reply.started":"2023-06-21T07:31:30.127099Z","shell.execute_reply":"2023-06-21T07:31:30.134832Z"},"jupyter":{"outputs_hidden":false}}
from torchvision import transforms
class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

# %% [code] {"execution":{"iopub.status.busy":"2023-06-21T07:31:30.805322Z","iopub.execute_input":"2023-06-21T07:31:30.80598Z","iopub.status.idle":"2023-06-21T07:31:30.821006Z","shell.execute_reply.started":"2023-06-21T07:31:30.805948Z","shell.execute_reply":"2023-06-21T07:31:30.819871Z"},"jupyter":{"outputs_hidden":false}}
import math
class AlignCollate(object):

    def __init__(self, imgH=64, imgW=64, keep_ratio_with_pad=False, contrast_adjust = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, images):

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size

                #### augmentation here - change contrast
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target = self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors
    
AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-21T07:31:31.472665Z","iopub.execute_input":"2023-06-21T07:31:31.473307Z","iopub.status.idle":"2023-06-21T07:31:31.479282Z","shell.execute_reply.started":"2023-06-21T07:31:31.473274Z","shell.execute_reply":"2023-06-21T07:31:31.478142Z"},"jupyter":{"outputs_hidden":false}}
def get_prediction_ocr(image):
    timg = AlignCollate_valid([image.convert("L")])
    preds = ocr_model(timg,"")
    preds_size = torch.IntTensor([preds.size(1)] * 1)
    _, preds_index = preds.max(2)
    preds_index = preds_index.view(-1)
    preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
    return preds_str[0]

# %% [code] {"papermill":{"duration":0.018581,"end_time":"2023-06-19T15:44:53.259546","exception":false,"start_time":"2023-06-19T15:44:53.240965","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-06-21T07:31:35.745086Z","iopub.execute_input":"2023-06-21T07:31:35.745446Z","iopub.status.idle":"2023-06-21T07:31:35.751726Z","shell.execute_reply.started":"2023-06-21T07:31:35.745414Z","shell.execute_reply":"2023-06-21T07:31:35.750406Z"},"jupyter":{"outputs_hidden":false}}
import re
def is_number(s: str):
    s = re.sub('[,$% ]', '', s)
    try:
        float(s)
        return True
    except ValueError:
        return False
def get_number(s):
    if type(s)==str:
        return float(re.sub('[,$% ]', '', s))
    else:
        return float(s)

def extract_text_from_box(box_ndarray):
    """
    Extract textual content from an image of a box using Optical Character Recognition (OCR).
    
    This function takes a file path to an image, converts it into an image object, 
    and then utilizes an OCR model to extract and return any textual content 
    contained within the image.
    
    Parameters:
    box_path (str): A file path string pointing to the image to be processed.
    
    Returns:
    str: The text extracted from the image.
    
    Note: 
    This function utilizes the `cv2.imread()` function to load the image, and 
    `get_prediction_ocr()` (assumed to be defined elsewhere) to perform the 
    text extraction. Ensure relevant dependencies are installed and imported.
    """
    # Convert the ndarray box to an image object
    # nd_array = cv2.imread(box_ndarray)
    
    # box_image = Image.fromarray(nd_array)

    box_image = Image.fromarray(box_ndarray)

    
    # Extract text using OCR
    extracted_text = get_prediction_ocr(box_image)
    
    return extracted_text


# print(extract_text_from_box("/kaggle/input/test-1/Weixin Image_20231014204722.png"))


