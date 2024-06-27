import numpy as np
import cv2
from config import cfg
import os
from loguru import logger
import datetime
from io import BytesIO
import time
import freetype
import copy

class MyImgUtil(object):
    def __init__(self, rescale_ratio = 1.0):
        self.rescale_ratio = None if rescale_ratio == 1.0 else rescale_ratio
        self.do_rescale = self.rescale_ratio is not None
        #text related
        self.ft_face = freetype.Face(cfg.TTF_PATH)
        self.text_size_ratio = cfg.TEXT_SIZE_RATIO
        self.text_size_det = cfg.TEXT_SIZE_DET
        self.text_font_color = cfg.TEXT_FONT_COLOR
        self.text_bg_color = cfg.TEXT_BG_COLOR
        self.text_bg_alpha = cfg.TEXT_BG_ALPHA
        self.margin_left = cfg.TEXT_MARGIN_LEFT
        self.margin_bot = cfg.TEXT_MARGIN_BOT
        self.blend_alpha = cfg.BLEND_ALPHA

    @logger.catch
    def merge_color(self, img, pos1, pos2):
        def _cal_cell(orig, color, alpha):
            t1 = (orig*(1-np.array(alpha)) + np.array(color)*np.array(alpha)).astype(np.uint8)
            t1[t1>255] = 255
            t1[t1<0] = 0
            return t1
        
        x1,y1 = int(pos1[0]),int(pos1[1])
        x2,y2 = int(pos2[0]),int(pos2[1])
        img[y1:y2,x1:x2,:] = _cal_cell(img[y1:y2,x1:x2,:],self.text_bg_color,self.text_bg_alpha)
        return img

    @logger.catch
    def blend_imgs(self, img1, img2, pos1, pos2):
        def _cal_cell(bg, fg, alpha):
            t1 = (bg * alpha + fg * (1-alpha)).astype(np.uint8)
            t1 = np.clip(t1, 0, 255)
            return t1
             
        x1,y1 = int(pos1[0]),int(pos1[1])
        x2,y2 = int(pos2[0]),int(pos2[1])
        img_out = img1.copy()
        #img_out.astype(np.float32)
        img_out[y1:y2,x1:x2,:] = _cal_cell(img_out[y1:y2,x1:x2,:],img2, self.blend_alpha)
        return img_out

    @logger.catch
    def blend_imgs_mask(self, img1, img2, mask):
        mask1 = np.expand_dims((1 - mask * self.blend_alpha),-1)
        img_out = img1 * mask1 + img2* (1-mask1)
        return img_out    
    

    @logger.catch
    def ft_text_batch(self, image, batch_pos, batch_text, batch_color):
        im_h, im_w, _ = image.shape
        text_size = self.text_size_det
        self.ft_face.set_char_size(text_size*64)
        metrics = self.ft_face.size
        ascender = metrics.ascender/64.0
        ypos = int(ascender)        
        for i in range(len(batch_pos)):
            batch_pos[i][1]+=ypos
        image = self.draw_string_batch(image,batch_pos,batch_text, batch_color)
        return image
 
    @logger.catch
    def draw_string_batch(self, img, batch_pos, batch_text, batch_color):
        pen = freetype.Vector()
        image = copy.deepcopy(img)
        im_num = len(batch_text)
        hscale = 1.0
        
        for i in range(im_num):
            x_pos = batch_pos[i][0]
            y_pos = batch_pos[i][1]
            text = batch_text[i]
            color = batch_color[i]

            prev_char = 0
            pen.x = x_pos << 6
            pen.y = y_pos << 6

            cur_pen = freetype.Vector()
            for cur_char in text:
                self.ft_face.load_char(cur_char)
                kerning = self.ft_face.get_kerning(prev_char, cur_char)
                pen.x += kerning.x
                slot = self.ft_face.glyph
                bitmap = slot.bitmap

                cur_pen.x = pen.x
                cur_pen.y = pen.y - slot.bitmap_top *64
                self.draw_ft_bitmap(image, bitmap, cur_pen, color)

                pen.x += slot.advance.x
                prev_char = cur_char
        return image

    @logger.catch
    def draw_text(self, image, txt_line, pos, rescale_ratio = None):
        im_h,im_w,_ = image.shape
        text_size = round(im_h / self.text_size_ratio)
        self.ft_face.set_char_size(text_size * 64)
        bg_pos1 = (pos[0],pos[1]-10)
        bg_pos2 = (min(pos[0] + int(text_size* len(txt_line)/1.5),im_w),pos[1]+text_size*1.4)
        img = self.merge_color(image, bg_pos1, bg_pos2)
        #img = image.copy()

        metrics = self.ft_face.size
        ascender = metrics.ascender/64.0
        ypos = int(ascender)
        img = self.draw_string(img,int(pos[0]+10),int(pos[1]+ypos), txt_line, self.text_font_color)
        #if not isinstance(text, unicode):
        #    text = text.decode('utf-8')
        #img = self.draw_string(image, pos[0], pos[1]+ypos, text, self.text_font_color)
        if rescale_ratio is not None:
            img = cv2.resize(img, None, fx=rescale_ratio, fy=rescale_ratio)
        return img

    @logger.catch
    def draw_string(self, img, x_pos, y_pos, text, color):
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6
        pen.y = y_pos << 6

        hscale = 1.0
        cur_pen = freetype.Vector()
        image = copy.deepcopy(img)
        for cur_char in text:
            self.ft_face.load_char(cur_char)
            kerning = self.ft_face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self.ft_face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top *64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char
        return image

    @logger.catch
    def draw_ft_bitmap(self, img, bitmap, pen, color):
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows
        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row*cols+col] != 0 and x_pos+col < img.shape[1] and y_pos+row < img.shape[0]:
                    img[y_pos + row][x_pos+col][0] = color[0]
                    img[y_pos + row][x_pos+col][1] = color[1]
                    img[y_pos + row][x_pos+col][2] = color[2]

            
    @logger.catch
    def draw_img(self, img, pred, down=16, base=0.0, enhance=1.2,roi=[]):
        has_rect_roi = False
        has_poly_roi = False
        if len(roi) == 4:
            has_rect_roi = True
            x1,y1,x2,y2 = roi
        elif len(roi) >= 6:
            has_poly_roi = True


        h,w,_ = img.shape
        img_data = img.copy()
        info = {'img':None, 't_sum':0, 'r_sum':0}        

        info['t_sum'] = np.sum(pred)
        if has_rect_roi:
            todo = pred[y1//down:y2//down,x1//down:x2//down]
            info['r_sum'] = np.sum(todo)
            todo = cv2.resize(todo, (x2-x1,y2-y1))
        elif has_poly_roi:
            n_p = len(roi)//2
            roi1 = np.array([[roi[2*x],roi[2*x+1]] for x in range(n_p)],dtype=np.int32)
            todo = cv2.resize(pred, (w,h))
            mask = np.zeros((h,w),dtype=np.float32)
            mask = cv2.fillPoly(mask, [roi1], (1.0,))
            todo = todo * mask
            info['r_sum'] = np.sum(todo)/down/down
        else:
            todo = pred
            todo = cv2.resize(todo, (w,h))
            
        todo = np.clip((todo+base) * enhance, 0, 1)
        todo = (todo * 255).astype(np.uint8)
        todo = cv2.applyColorMap(todo, cv2.COLORMAP_JET)

    
        if has_rect_roi:
            blended = self.blend_imgs(img_data, todo, (x1,y1), (x2,y2))
            cv2.rectangle(blended, (x1,y1),(x2,y2), (0,0,255), 2)
            txt_line = f'选定区域内人数: {info["r_sum"]:.2f}    全图总人数: {info["t_sum"]:.2f}'
            #cv2.putText(blended,f'roi: {info["r_sum"]:.2f}, total: {info["t_sum"]:.2f}',(100,200),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        elif has_poly_roi:
            blended = self.blend_imgs_mask(img_data, todo, mask)
            cv2.polylines(blended, [roi1], True, (0,0,255), 2)
            txt_line = f'选定区域内人数: {info["r_sum"]:.2f}    全图总人数: {info["t_sum"]:.2f}' 
        else:
            blended = self.blend_imgs(img_data, todo, (0,0), (w,h))
            txt_line = f'全图总人数: {info["t_sum"]}'
            #cv2.putText(blended,f'total: {info["t_sum"]:.2f}',(100,200),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        blended = self.draw_text(blended, txt_line, (100,200))
        if self.do_rescale:
            blended = cv2.resize(blended,None,fx=self.rescale_ratio,fy=self.rescale_ratio)
        info['img'] = blended
        return info
        

if __name__ == '__main__':
    import sys
    im_path = sys.argv[1] 
    img = cv2.imread(im_path)
    h,w,_ = img.shape
    ut = MyImgUtil()
    pred = np.random.randn(h//16, w//16)
    info = ut.draw_img(img, pred, roi = [230,700,2360,420,2559,628,2559,880,230,1400])
    print(f't_sum={info["t_sum"]}, r_sum={info["r_sum"]}')
    cv2.imwrite('dump_sample.jpg',info['img'])
