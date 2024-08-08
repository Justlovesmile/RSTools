import os
# 解除opencv对图像尺寸的限制，需添加在`import cv2`之前；
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from rstools.utils.image import read_image, resize_image


def xywha2xy4(xywha):  # a represents the angle(degree), clockwise, a=0 along the X axis
    x, y, w, h, a = xywha
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    # a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def drawBbox(img,locations,color,thickness=1):
    x1,y1,x2,y2,x3,y3,x4,y4 = locations
    cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),color,thickness)
    cv2.line(img,(int(x2),int(y2)),(int(x3),int(y3)),color,thickness)
    cv2.line(img,(int(x3),int(y3)),(int(x4),int(y4)),color,thickness)
    cv2.line(img,(int(x4),int(y4)),(int(x1),int(y1)),color,thickness)


def drawPoly(img,points,color,thickness=1):
    points = np.array([points], dtype = np.int32)
    cv2.polylines(img,points,isClosed=True,color=color,thickness=thickness)  


def cv2ImgAddText(img, text_info, text_lang='en', font_path=None, text_line=False):
    if text_lang=='en':
        for (text, position, start_point, text_size, text_thickness, text_color) in text_info:
            cv2.putText(img,text,position,cv2.FONT_HERSHEY_SIMPLEX,text_size,text_color,text_thickness)
    elif text_lang=='zh-cn':
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            isBGR = True
        draw = ImageDraw.Draw(img)
        for (text, position, start_point, text_size, text_thickness, text_color) in text_info:
            fontStyle = ImageFont.truetype(font_path, text_size, encoding="utf-8")
            text_color = text_color[::-1] if isBGR else text_color
            draw.text(position, text, text_color, font=fontStyle)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    else:
        return img
    if text_line:
        for (text, position, start_point, text_size, text_thickness, text_color) in text_info:
            cv2.line(img, start_point, (position[0],position[1]+9), text_color, thickness=1)
    return img

def isTextOverlap(new_text, text_lists, anns=None, box_key='bbox', area_thresh=10000):
    def getTextBox(text, charpix=18):
        length = len(text[0])*charpix
        xmin, ymin = text[1]
        xmax, ymax = xmin+length, ymin+charpix
        return xmin, ymin, xmax, ymax

    nxmin, nymin, nxmax, nymax = getTextBox(new_text)
    for text in text_lists:
        bxmin, bymin, bxmax, bymax = getTextBox(text)
        minx, miny = max(nxmin, bxmin), max(nymin, bymin)
        maxx, maxy = min(nxmax, bxmax), min(nymax, bymax)
        if not (minx > maxx or miny > maxy):
            return True
    if anns:
        for ann in anns:
            if len(ann[box_key]) == 5:
                [[x1,y1,x2,y2,x3,y3,x4,y4]] = xywha2xy4(ann[box_key]).reshape(1, -1).tolist()
                bxmin, bymin = min([x1,x2,x3,x4]), min([y1,y2,y3,y4])
                bxmax, bymax = max([x1,x2,x3,x4]), max([y1,y2,y3,y4])
            elif len(ann[box_key]) == 8:
                [x1,y1,x2,y2,x3,y3,x4,y4] = ann[box_key]
                bxmin, bymin = min([x1,x2,x3,x4]), min([y1,y2,y3,y4])
                bxmax, bymax = max([x1,x2,x3,x4]), max([y1,y2,y3,y4])
            elif len(ann[box_key]) == 4:
                [bxmin,bymin,w,h] = ann[box_key]
                bxmax, bymax = bxmin+w, bymin+h
            if (bxmax-bxmin)*(bymax-bymin) < area_thresh:
                minx,miny = max(nxmin,bxmin), max(nymin,bymin)
                maxx, maxy = min(nxmax,bxmax), min(nymax,bymax)
                if not(minx > maxx or miny > maxy):
                    return True
    return False


def adjust_text(xlist,ylist,text,text_info,anns,box_key):
    start_points = []
    # top point
    sy,sx = min(ylist),xlist[ylist.index(min(ylist))]
    start_points.append((sx,sy))
    ylist.pop(ylist.index(sy)),xlist.pop(xlist.index(sx))
    # right point
    sx,sy = max(xlist),ylist[xlist.index(max(xlist))]
    start_points.append((sx,sy))
    ylist.pop(ylist.index(sy)),xlist.pop(xlist.index(sx))
    # down point
    sy,sx = max(ylist),xlist[ylist.index(max(ylist))]
    start_points.append((sx,sy))
    ylist.pop(ylist.index(sy)),xlist.pop(xlist.index(sx))
    # left point
    start_points.append((xlist.pop(),ylist.pop()))
    # move direction
    move_directs = [(-1,-1),(1,-1),(1,1),(-1,1)]
    # init point
    sx, sy = start_points[0]
    tx, ty = sx + 18, sy - 18
    iteridx = 0
    # iteration
    while isTextOverlap([text, (tx,ty)], text_info, anns, box_key):
        sx, sy = start_points[iteridx%4]
        tx, ty = sx + move_directs[iteridx%4][0]*18, sy + move_directs[iteridx%4][1]*18
        tx += move_directs[iteridx%4][0] * (iteridx//4+1)
        ty += move_directs[iteridx%4][1] * (iteridx//4+1)
        iteridx += 1
    return tx,ty,sx,sy


def visual_dataset(
        img_path,
        anns,
        color=(0,255,0),
        color_key='category_id',
        box_key='bbox',
        seg_key='segmentation',
        thickness=2,
        ratio=1.0,
        vis_mode=None, # ['jupyter', 'pil', 'opencv']
        save_path=None,
        put_text=False,
        text_key='name',
        text_lang='en', # ['en','zh-cn']
        text_scale=None,
        text_thickness=None,
        text_adjust=False,
        font_path='../src/Chinese.ttf',
        show_box=True,
        show_seg=True,
    ):
    # Check
    assert img_path and os.path.exists(img_path), f"[ERROR] Can not find img path: {img_path}"
    if text_lang == 'zh-cn':
        assert font_path and os.path.exists(font_path), f"[ERROR] Can not find font path: {font_path}"
    if put_text:
        text_info = []
    # read image
    img = read_image(img_path)
    # data parse
    for aidx, ann in enumerate(anns):
        # color
        if 'color' in ann.keys():
            c = ann['color']
        elif type(color)==dict:
            c = color[ann[color_key]]
        elif type(color)==list:
            c = color[aidx]
        else:
            c = color
        
        # draw bounding box
        if show_box and box_key in ann.keys():
            if len(ann[box_key]) == 5:
                [[x1,y1,x2,y2,x3,y3,x4,y4]] = xywha2xy4(ann[box_key]).reshape(1, -1).tolist()
            elif len(ann[box_key]) == 8:
                [x1,y1,x2,y2,x3,y3,x4,y4] = ann[box_key]
            elif len(ann[box_key]) == 4:
                [xmin,ymin,w,h] = ann[box_key]
                xmax, ymax = xmin+w, ymin+h
                [x1,y1,x2,y2,x3,y3,x4,y4] = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]
            drawBbox(img,[x1,y1,x2,y2,x3,y3,x4,y4],c,thickness)

        # draw segmentation
        if show_seg and seg_key in ann.keys():
            points = []
            if len(ann[seg_key])==1:
                for i in range(len(ann[seg_key][0]))[::2]:
                    x = [ann[seg_key][0][i], ann[seg_key][0][i+1]]
                    points.append(x)
            elif len(ann[seg_key])>=4 and ann[seg_key][0]==ann[seg_key][-1]:
                points = ann[seg_key]
            drawPoly(img,points,c,thickness)
        
        # add text annotation
        if put_text:
            if type(text_key)==list:
                text = ':'.join([str(ann[k]) for k in text_key])
            else:
                text = str(ann[text_key]) #.strip('_其它')
            if not text_scale:
                text_scale = 0.75 if text_lang=='en' else 18
            if not text_thickness:
                text_thickness = 2
            ylist = [y1,y2,y3,y4]
            xlist = [x1,x2,x3,x4]
            if text_adjust:
                tx,ty,sx,sy = adjust_text(xlist,ylist,text,text_info,anns,box_key)
            else:
                sy = min(ylist)
                sx = xlist[ylist.index(sy)]
                ty, tx = sy - 18, sx + 18
            text_info.append((text, (int(tx),int(ty)), (int(sx),int(sy)), text_scale, text_thickness, c))

    if put_text:
        img = cv2ImgAddText(img, text_info, text_lang, font_path, text_adjust)

    img = resize_image(img,ratio)
    if vis_mode == 'jupyter':
        display(Image.fromarray(img.astype('uint8')))
    elif vis_mode == 'opencv':
        cv2.imshow('visual',img)
        cv2.waitKey(0)
    elif vis_mode == 'pil':
        Image.fromarray(img.astype('uint8')).show(title='visual')
    if save_path:
        cv2.imwrite(save_path,img)
    return img


def random_color(num):
    color = []
    for _ in range(num):
        color.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    return color


def get_palette():
    return [(255, 255, 0), (45, 89, 255), (0, 0, 142), (0, 165, 120), (220, 20, 60), 
            (255, 250, 205), (255, 99, 164), (0, 226, 252), (59, 105, 106), (175, 116, 175), 
            (208, 195, 210), (95, 32, 0), (0, 143, 149), (0, 220, 176), (0, 228, 0), 
            (174, 57, 255), (78, 180, 255), (220, 220, 0), (255, 0, 0), (95, 54, 80), 
            (72, 0, 118), (0, 0, 230), (0, 51, 153), (119, 0, 170), (147, 116, 116), 
            (100, 170, 30), (79, 210, 114), (109, 63, 54), (171, 134, 1), (106, 0, 228), 
            (151, 0, 95), (165, 42, 42), (0, 182, 199), (0, 0, 70), (0, 125, 92), 
            (110, 129, 133), (65, 70, 15), (74, 65, 105), (250, 0, 30), (255, 128, 0), 
            (166, 74, 118), (255, 193, 193), (0, 0, 255), (255, 179, 240), (0, 255, 255), 
            (209, 0, 151), (128, 76, 255), (127, 167, 115), (0, 82, 0), (134, 134, 103), 
            (92, 0, 73), (3, 95, 161), (110, 76, 0), (191, 162, 208), (119, 11, 32), 
            (163, 255, 0), (255, 0, 255), (9, 80, 61), (0, 60, 100), (145, 148, 174), 
            (179, 0, 194), (197, 226, 255), (183, 130, 88), (0, 139, 139), (188, 208, 182), 
            (201, 57, 1), (120, 166, 157), (255, 109, 65), (178, 90, 62), (84, 105, 51), 
            (199, 100, 0), (147, 186, 208), (142, 108, 45), (227, 255, 205), (255, 77, 255), 
            (189, 183, 107), (5, 121, 0), (250, 170, 30), (0, 80, 100), (255, 208, 186), 
            (207, 138, 255), (196, 172, 0), (0, 255, 0), (0, 0, 192), (138, 43, 226), 
            (133, 129, 255), (182, 182, 255), (153, 69, 1), (166, 196, 102), (246, 0, 122), 
            (209, 99, 106), (219, 142, 185), (130, 114, 135), (174, 255, 243)]