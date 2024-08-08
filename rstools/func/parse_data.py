import os
import json
import itertools
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

from rstools.utils.file import get_file_from_this_dir
from rstools.utils.visual import visual_dataset, get_palette


IMGEXT = ['.tif','.TIF','.tiff','.TIFF','.jpg','.JPG','.jpeg','.JPEG',
          '.png','.PNG','.gif','.GIF','.webp','.WEBP','.svg', '.SVG',
          '.bmp', '.BMP', '.raw', '.RAW']


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class ParseDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        pal = get_palette()
        self.palette = dict(zip(list(range(1,len(pal)+1)),pal))
        self.createIndex()

    def createIndex(self):       
        self.anns,self.cats,self.imgs = dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.imgToAnns[ann['image_id']].append(ann)
                self.anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                self.catToImgs[ann['category_id']].append(ann['image_id'])

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def getImgNum(self):
        return len(self.dataset['images'])
    
    def getAnnNum(self):
        return len(self.dataset['annotations'])
    
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, img_id, **kwargs):
        vis_img = visual_dataset(
            self.imgs[img_id]['file_name'],
            self.imgToAnns[img_id],
            **kwargs
        )
        return vis_img


class ParseCOCO(ParseDataset):
    def __init__(self, jsonpath):
        with open(jsonpath, 'r') as f:
            dataset = json.load(f)
        ParseDataset.__init__(self,dataset=dataset)


class ParseDOTA(ParseDataset):
    def __init__(self, imgdir, labeldir, ext=None):
        self.imgdir = imgdir
        self.labeldir = labeldir
        self.samedir = (os.path.abspath(imgdir) == os.path.abspath(labeldir))
        self.ext = ext if ext else IMGEXT
        ParseDataset.__init__(self,dataset=self.parse_dataset())
        
    def parse_dataset(self):
        dataset = dict(images=[],annotations=[],categories=[])
        wordnames = []
        imgpaths = get_file_from_this_dir(self.imgdir, ext=self.ext)
        ann_idx = 0
        for idx,imgpath in enumerate(imgpaths):
            dataset['images'].append(dict(id=idx+1,file_name=imgpath))
            imgfile = os.path.basename(imgpath)
            _, imgext = os.path.splitext(imgfile)
            txtfile = imgfile.replace(imgext, '.txt')
            if self.samedir:
                txtpath = imgpath.replace(imgfile, txtfile)
            else:
                txtpath = os.path.join(self.labeldir, txtfile)
            if not os.path.exists(txtpath):
                continue
            with open(txtpath,'r') as f:
                lines = f.readlines()
            for line in lines:
                splitline = line.strip().split()
                obj = {}
                if len(splitline) < 9:
                    continue
                [x1,y1,x2,y2,x3,y3,x4,y4] = list(map(float,splitline[:8]))
                obj['bbox'] = [x1,y1,x2,y2,x3,y3,x4,y4]
                obj['name'] = splitline[8]
                if obj['name'] not in wordnames:
                    wordnames.append(obj['name'])
                    dataset['categories'].append(dict(id=len(wordnames),name=obj['name'],supercategory=obj['name']))
                if len(splitline) == 9:
                    obj['difficult'] = 0
                elif len(splitline) == 10:
                    obj['difficult'] = int(splitline[9])
                ann_idx += 1
                dataset['annotations'].append(dict(category_id=wordnames.index(obj['name'])+1,
                                                label=obj['name'],
                                                iscrowd=obj['difficult'],
                                                bbox=obj['bbox'],
                                                image_id=idx+1,
                                                id=ann_idx))  
        return dataset
    
class ParseRSLabel(ParseDataset):
    def __init__(self, imgdir, labeldir, ext=None):
        self.imgdir = imgdir
        self.labeldir = labeldir
        self.samedir = (os.path.abspath(imgdir) == os.path.abspath(labeldir))
        self.ext = ext if ext else IMGEXT
        ParseDataset.__init__(self,dataset=self.parse_dataset())

    def parse_dataset(self):
        dataset = dict(images=[],annotations=[],categories=[])
        wordnames = []
        imgpaths = get_file_from_this_dir(self.imgdir, ext=self.ext)
        ann_idx = 0
        for idx,imgpath in enumerate(imgpaths):
            dataset['images'].append(dict(id=idx+1,file_name=imgpath))
            imgfile = os.path.basename(imgpath)
            _, imgext = os.path.splitext(imgfile)
            jsonfile = imgfile.replace(imgext, '.json')
            if self.samedir:
                jsonpath = imgpath.replace(imgfile, jsonfile)
            else:
                jsonpath = os.path.join(self.labeldir, jsonfile)
            if not os.path.exists(jsonpath):
                continue
            with open(jsonpath,'r', encoding='utf-8') as f:
                data = json.load(f)
            geotrans = data.get('geoTrans','geo not exist')
            for shape in data['shapes']:
                pts = shape['points']
                pts_4 =[[0,0],[0,0],[0,0],[0,0]]
                if len(pts) == 8:
                    pts_4[0] = pts[0]
                    pts_4[1] = pts[2]
                    pts_4[2] = pts[4]
                    pts_4[3] = pts[6] 
                elif len(pts) == 2:
                    pts_4[0] = [pts[0][0],pts[0][1]]                 
                    pts_4[1] = [pts[1][0],pts[0][1]]                 
                    pts_4[2] = [pts[1][0],pts[1][1]]
                    pts_4[3] = [pts[0][0],pts[1][1]]  
                elif len(pts) == 4:
                    pts_4[0] = pts[0]
                    pts_4[1] = pts[1]
                    pts_4[2] = pts[2]
                    pts_4[3] = pts[3]                     
                label = shape['label']
                if label not in wordnames:
                    wordnames.append(label)
                    dataset['categories'].append(dict(id=len(wordnames),name=label,supercategory=label))
                difficult = 0
                pbox_geo = [0] * 8
                dTemp = geotrans[1] * geotrans[5] - geotrans[2] * geotrans[4]

                pbox_geo[0] = (geotrans[5] * (pts_4[0][0] - geotrans[0]) - geotrans[2] * (pts_4[0][1] - geotrans[3])) / dTemp 
                pbox_geo[1] = (geotrans[1] * (pts_4[0][1] - geotrans[3]) - geotrans[4] * (pts_4[0][0]- geotrans[0])) / dTemp 

                pbox_geo[2] = (geotrans[5] * (pts_4[1][0] - geotrans[0]) - geotrans[2] * (pts_4[1][1] - geotrans[3])) / dTemp 
                pbox_geo[3] = (geotrans[1] * (pts_4[1][1] - geotrans[3]) - geotrans[4] * (pts_4[1][0] - geotrans[0])) / dTemp 

                pbox_geo[4] = (geotrans[5] * (pts_4[2][0] - geotrans[0]) - geotrans[2] * (pts_4[2][1] - geotrans[3])) / dTemp 
                pbox_geo[5] = (geotrans[1] * (pts_4[2][1] - geotrans[3]) - geotrans[4] * (pts_4[2][0] - geotrans[0])) / dTemp 

                pbox_geo[6] = (geotrans[5] * (pts_4[3][0] - geotrans[0]) - geotrans[2] * (pts_4[3][1] - geotrans[3])) / dTemp 
                pbox_geo[7] = (geotrans[1] * (pts_4[3][1] - geotrans[3]) - geotrans[4] * (pts_4[3][0] - geotrans[0])) / dTemp 

                bbox = [pbox_geo[0],pbox_geo[1],pbox_geo[2],pbox_geo[3],pbox_geo[4],pbox_geo[5],pbox_geo[6],pbox_geo[7]]
                ann_idx += 1
                dataset['annotations'].append(dict(category_id=wordnames.index(label)+1,
                                                    label=label,
                                                    iscrowd=difficult,
                                                    bbox=bbox,
                                                    image_id=idx+1,
                                                    id=ann_idx))
        return dataset


if __name__ == "__main__":
    import sys

    imgdir, labeldir, savedir = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"[INFO] Input Argv: {sys.argv}")
    os.makedirs(savedir, exist_ok=True)

    dataset = ParseRSLabel(imgdir,labeldir)

    for i in range(dataset.getImgNum()):
        imgname = os.path.basename(dataset.imgs[i+1]['file_name'])
        print(f"[INFO] Visualing {i+1}-th image: {dataset.imgs[i+1]['file_name']}")
        dataset.showAnns(i+1,color=dataset.palette,save_path=os.path.join(savedir,os.path.splitext(imgname)[0]+'_vis.jpg'),
                        put_text=True, text_lang='zh-cn', font_path='../src/SimHei.ttf',
                        text_key='label', vis_mode='none', text_adjust=True)