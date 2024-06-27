import torch
import os
import numpy as np
from models.vgg_c import vgg19_trans
from glob import glob
from torchvision import transforms
from PIL import Image
import argparse
import cv2
from shutil import copy
from loguru import logger
from vis_utils import MyImgUtil
import onnxsim
import onnx
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--weights', type=str,required=True,help='checkpoint path')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--save_path',type=str,default='model.onnx',help='output path')
    parser.add_argument('--opset',type=int,default=15,help='opset version')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    torch.backends.cudnn.benchmark = False
   
    device = torch.device('cuda')
    model = vgg19_trans()
    model.load_state_dict(torch.load(args.weights, device))
    #model.to(device)
    model.eval()

    
    dummy_input = torch.randn(1,3,1728,3072)
    #dummy_input = torch.randn(1,3,1440,2560)
    #dummy_input = torch.randn(1,3,720,1280)
    #dummy_input = torch.randn(1,3,1152,2048)
    #dummy_input.to(device)
    torch.onnx.export(model, dummy_input, args.save_path, verbose=False, input_names=['data'], output_names=['output'], opset_version=args.opset)
    
    model = onnx.load(args.save_path)
    onnx.checker.check_model(model)
    sim_model,check = onnxsim.simplify(model)
    assert check, 'assert check failed'
    onnx.save(sim_model, args.save_path)

