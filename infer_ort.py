import os
import numpy as np
from glob import glob
from PIL import Image
import argparse
import cv2
from shutil import copy
from loguru import logger
from vis_utils import MyImgUtil
import onnxruntime as ort

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default=r'E:\Dataset\Counting\UCF-Train-Val-Test\test',
                        help='training data directory')
    parser.add_argument('--model-path', default='model/best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--outdir',type=str,default='weights',help='output dir')
    parser.add_argument('--roi',type=int,nargs='+',default=[],help='roi')
    #parser.add_argument('--fp16',action='store_true',help='use fp16')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.outdir
   
    painter = MyImgUtil() 
    roi_candi = args.roi
    roi_tl = None
    roi_br = None
    roi = []
    if len(roi_candi) >=6:
        use_roi = True
        roi = roi_candi
    elif len(roi_candi)!=4:
        use_roi = False
    elif roi_candi[0] <0 or roi_candi[2] < 0:
        use_roi = False
    elif roi_candi[0] < roi_candi[2] and roi_candi[1] < roi_candi[3]:
        use_roi = True
        x1,y1 = roi_candi[0],roi_candi[1]
        x2,y2 = roi_candi[2],roi_candi[3]
        roi = roi_candi
    else:
        use_roi = False

    save_dir_d = os.path.join(save_dir, 'density')
    if not os.path.exists(save_dir_d):
        os.makedirs(save_dir_d)

    save_dir_viz = os.path.join(save_dir, 'vis')
    if not os.path.exists(save_dir_viz):
        os.makedirs(save_dir_viz)

    #trans = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.485, 0.456, 0.406],
    #                         [0.229, 0.224, 0.225])
    #])

    im_list = glob(os.path.join(args.data_dir, '*.jpg'))
    sess_opt = ort.SessionOptions()
    session = ort.InferenceSession(args.model_path, sess_options=sess_opt,
        providers=[('CUDAExecutionProvider',{'device_id':args.device,'cudnn_conv_use_max_workspace':1}),'CPUExecutionProvider'])

    
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
    export = True
    alpha = 0.5
    
    model_input = session.get_inputs()[0]
    inp_name = model_input.name
    inp_dtype = model_input.type
    logger.info(f'inp_name={inp_name}, inp_dtype={inp_dtype}')
    is_fp16 = 'float16' in inp_dtype    


    if True:
        for im_path in im_list:
            #keypoints = np.load(gd_path)
            name = os.path.basename(im_path).split('.')
            # print(name)
            img = Image.open(im_path).convert('RGB')
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            if h != 1440 or w != 2560:
                img_resize = cv2.resize(img_np,(2560,1440), cv2.INTER_LINEAR)
            else:
                img_resize = img_np.copy()
            img_1 = img_resize.astype(np.float32)/255
            img_1 = (img_1 - mean)/std
            img_1 = np.transpose(img_1, (2,0,1))
            img_1 = np.expand_dims(img_1,0) 
            
            if is_fp16:
                img_1 = img_1.astype(np.float16)           
 
            inputs = {inp_name: img_1}
            outputs = session.run(None, inputs)[0].astype(np.float32)

            
            
            logger.info(f'outputs.shape={outputs.shape}')
            logf = '{}.jpg'.format(name[0])
            outputs = outputs[0][0]
            logger.info(f'outputs.shape={outputs.shape}')
            #np.save(os.path.join(save_dir_d, name[0]+'.npy'), outputs)

            info = painter.draw_img(img_resize, outputs, base=0.1,enhance=1.5,roi=roi)
            logger.info(f'name={name[0]}, total_sum={info["t_sum"]}, roi_sum={info["r_sum"]}')
            if h != 1440 or w!=2560:            
                blended = cv2.resize(info['img'],(w,h),cv2.INTER_LINEAR)
            else:
                blended = info['img']
            '''
            # outputs = outputs / np.max(outputs) * 255
            t_sum = np.sum(outputs)
            if not use_roi:
                logger.info(f'name={name[0]}, total_sum={t_sum}')
            else:
                back = np.zeros([h//16,w//16])
                out_roi = outputs[y1//16:y2//16,x1//16:x2//16]
                back[y1//16:y2//16,x1//16:x2//16] = out_roi
                r_sum = np.sum(out_roi)
                logger.info(f'name={name[0]}, roi_sum={r_sum}, total_sum={t_sum}')
                outputs = back
            outputs = cv2.resize(outputs, (w, h)) / 1.0
            outputs = np.clip(outputs,0.0,1.0)
            logger.info(f'expand_sum={np.sum(outputs)}, scaled={np.sum(outputs)/256}') 
            outputs = outputs * 255
            outputs = outputs.astype(np.uint8)
            outputs = cv2.applyColorMap(outputs, cv2.COLORMAP_JET)
            
        
            img_fl = img_np.astype('float32')
            out_fl = outputs.astype('float32')
            blended = cv2.addWeighted(img_fl,alpha,out_fl,1-alpha,0)
            blended = cv2.convertScaleAbs(blended)
            if use_roi:
                cv2.rectangle(blended, (x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(blended,f'roi: {r_sum:.2f}, total: {t_sum:.2f}',(100,200),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
            else:
                cv2.putText(blended,f'total: {t_sum:.2f}',(100,200),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
            '''    
        
            # copy(im_path,
            #      im_path.replace(os.path.dirname(im_path), save_dir_viz))
            # cv2.imwrite(os.path.join(save_dir_viz,
            #                          im_path.replace('.JPG', logf+'_d.jpg')), outputs)
            cv2.imwrite(os.path.join(save_dir_viz, logf), blended)

