import os
import json
import tqdm

from rstools.utils.file import get_file_from_this_dir, custom_basename
from rstools.utils.poly import parse_dota_poly2, polygonToRotRectangle
from rstools.utils.image import read_image
from rstools.func.img_split import ImgSplitBase, OnlyImgSplitBase

wordname_dotav1 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
wordname_dotav15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                'container-crane']
wordname_dotav2 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
                'container-crane', 'airport', 'helipad']

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2', ext='.png', save=True):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = get_file_from_this_dir(labelparent, ext='.txt')
        for file in tqdm.tqdm(filenames[::-1]):
            basename = custom_basename(file)
            imagepath = os.path.join(imageparent, basename + ext)
            try:
                img = read_image(imagepath)
                height, width = img.shape[:2]
            except Exception as e:
                print("[ERROR] Can not read the shape of img:", imagepath, f", due to {e}")

            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    #print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                
                # b = np.array(obj['poly']).astype(np.int32).reshape(4,2)
                # rbox = cv2.minAreaRect(b)
                # xcenter, ycenter, width, height, angle = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]

                rotated_box = polygonToRotRectangle(obj['poly'])
                xcenter, ycenter, width, height, angle = rotated_box
                single_obj['bbox'] = xcenter, ycenter, width, height, angle
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        if save:
            json.dump(data_dict, f_out)
        else:
            return data_dict

def DOTA2COCOTest(srcpath, destfile, cls_names, ext='.png', save=True):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = get_file_from_this_dir(imageparent, ext=ext)
        for file in tqdm.tqdm(filenames):
            basename = custom_basename(file)
            imagepath = os.path.join(imageparent, basename + ext)
            try:
                img = read_image(imagepath)
                height, width = img.shape[:2]
            except Exception as e:
                print("[ERROR] Can not read the shape of img:", imagepath, f", due to {e}")

            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        if save:
            json.dump(data_dict, f_out)
        else:
            return data_dict

def split_dota2coco_format(
        imgpath,
        labelpath,
        dstpath, 
        wordname,
        difficult='-1',
        dstdir='split', 
        subsize=800, 
        gap=200,
        thresh=0.7,
        num_process=20, 
        multi_scale=(1.0,), 
        only_image=False,
        padding=True,
        ext='.png',
        saveext='.png',
        filter_empty=False,
        save_json=False,
    ):
    if not os.path.exists(os.path.join(dstpath, dstdir)):
        os.makedirs(os.path.join(dstpath, dstdir))

    if not only_image:
        split_train = ImgSplitBase(
                        imgpath,
                        labelpath,
                        os.path.join(dstpath, dstdir),
                        gap=gap,
                        subsize=subsize,
                        thresh=thresh,
                        num_process=num_process,
                        padding=padding,
                        ext=ext,
                        saveext=saveext,
                        filter_empty=filter_empty,
                        difficult=difficult
                        )
        for ms in multi_scale:
            split_train.splitdata(ms)
        if save_json:
            DOTA2COCOTrain(os.path.join(dstpath, dstdir), os.path.join(dstpath, dstdir, f'trainval{subsize}_annotations.json'), wordname, difficult=difficult, ext=saveext)
    else:
        split_test = OnlyImgSplitBase(imgpath,
                        os.path.join(dstpath, dstdir),
                        gap=gap,
                        subsize=subsize,
                        num_process=num_process,
                        padding=padding,
                        ext=ext,
                        saveext=saveext
                        )
        for ms in multi_scale:
            split_test.splitdata(ms)
        if save_json:
            DOTA2COCOTest(os.path.join(dstpath, dstdir), os.path.join(dstpath, dstdir, f'test{subsize}_images.json'), wordname, ext=saveext)
    
if __name__ == "__main__":
    import sys
    imgpath, labelpath, dstpath = sys.argv[1], sys.argv[2], sys.argv[3]

    split_dota2coco_format(
        imgpath,
        labelpath,
        dstpath,
        wordname=['ship', 'plane'],
        difficult='2',
        dstdir='Data_1024_512',
        subsize=1024,
        gap=512,
        num_process=16,
        multi_scale=(1.0,),
        only_image=False,
        padding=True,
        ext='.tiff',
        saveext='.png',
        filter_empty=True,
        save_json=False
    )