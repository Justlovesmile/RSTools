import gc
import os
# 解除opencv对图像尺寸的限制，需添加在`import cv2`之前；
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import tqdm
import math
import numpy as np
from functools import partial
from multiprocessing import Pool
from osgeo import gdal, osr, gdalconst


def resize_image(img, scale=1.0):
    """等比例缩放图像
    Args:
        img: 输入的图像
        scale: 缩放比例
    Returns:
        img: 缩放后的图像
    """
    if (scale != 1.0):
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def strentch_img(imgarr, mode='linear', ntype=np.uint8, min_bias=1e-8, drange=255):
    """将图像拉伸至0-255范围
    Args:
        imgarr: 输入的图像
        mode: 选择拉伸方法['linear', 'plinear', 'olinear', 'alpha', 'gaussian',
              'equalization', 'sqrt', 'log1p', 'square']
        ntype: 输出的图像数据类型
        min_bias: 分母最小值
        drange: 拉伸范围
    Returns:
        imgarr: 拉伸后的图像
    """
    if mode == "linear" or mode == 1:  # Linear Stretch
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
    elif mode == "plinear" or mode == 2:  # Percent Linear Stretch
        percent = 2
        down, up = np.percentile(imgarr_vec, (percent, 100 - percent))
        imgarr = np.clip(imgarr, a_min=down, a_max=up)
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
    elif mode == "olinear" or mode == 3:  # Optimized Linear Stretch
        min_adjust_percent = 0.1
        max_adjust_percent = 0.5
        imgarr_vec = imgarr.ravel()
        imgarr_vec = imgarr_vec[np.where(imgarr_vec>0)] # 去除黑边
        down, up = np.percentile(imgarr_vec, (2.5, 99))
        del imgarr_vec
        black = down - min_adjust_percent * (up - down)
        white = up + max_adjust_percent * (up - down)
        imgarr = (imgarr - black)/(white - black + min_bias) * drange
        imgarr = np.clip(imgarr, 0, drange)
    elif mode == "alpha" or mode == 4:  # Alpha Stretch
        alpha = 2.5
        mean = np.mean(imgarr)
        imgarr = np.clip((imgarr/mean)**alpha * drange, 0, drange)
    elif mode == "gaussian" or mode == 5:  # Gaussian Stretch
        mean = np.mean(imgarr)
        std = np.std(imgarr)
        imgarr = np.clip(imgarr, a_min=(mean - (3 * std)), a_max=(mean + (3 * std)))
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
    elif mode == "equalization"  or mode == 6:  # Equalization Stretch
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
        if len(imgarr.shape) == 3:  # C,H,W
            for i in range(imgarr.shape[0]):
                imgarr[i,:,:] = cv2.equalizeHist(imgarr[i,:,:].astype(np.uint8))
        else:
            imgarr = cv2.equalizeHist(imgarr.astype(np.uint8))
    elif mode == "sqrt"  or mode == 7:  # Square Root Stretch
        imgarr = np.sqrt(imgarr)
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
    elif mode == "log1p"  or mode == 8:  # Log Stretch
        imgarr = np.log1p(imgarr)
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
    elif mode == "square"  or mode == 9:  # Square Stretch
        imgarr = np.square(imgarr)
        imgarr = (imgarr - np.min(imgarr))/(np.max(imgarr) - np.min(imgarr) + min_bias) * drange
    return imgarr.astype(ntype)


def read_RSTif(file_path, logger=False, return_info=False, norm=True, norm_mode=1):
    """使用GDAL读取栅格矩阵
    Args:
        file_path: 输入图像路径
        logger: 是否输出文件属性信息
        return_info: 是否返回文件属性信息
        norm: 是否对图像拉伸并转为RGB
        norm_mode: 拉伸方法选择
    Returns:
        imgtif,EPSGCode,im_proj,im_geotrans: 如果return_info为True
        imgtif: 如果return_info为False
    """
    dataset = gdal.Open(file_path)    # 打开文件
    if dataset == None:
        print(f"[WARNING] Can not open file: {file_path}!")
        return (None, None, None, None) if return_info else None
    im_width = dataset.RasterXSize    # 栅格矩阵的列数
    im_height = dataset.RasterYSize   # 栅格矩阵的行数
    imgtif = dataset.ReadAsArray(0, 0, im_width, im_height) # 将数据写成数组，对应栅格矩阵
    
    im_bands = dataset.RasterCount           # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj= dataset.GetProjection()         # 地图投影信息
    proj = osr.SpatialReference(im_proj)
    EPSGCode = proj.GetAttrValue('AUTHORITY',1)
    if logger:
        print(f"[INFO] IMG_W: {im_width}, IMG_H: {im_height}, IMG_B: {im_bands},"
              f"DTYPE: {imgtif.dtype}, EPSG_Code: {EPSGCode}({type(EPSGCode)}),"
              f"GEOTRANS: {im_geotrans}, PROJ: {im_proj}")
    
    if norm: # TIF to RGB
        if len(imgtif.shape) == 3:  # 多通道图像
            imgtif = imgtif[:3, :, :]
            imgtif = strentch_img(imgtif, norm_mode, np.uint8)
            imgtif = imgtif.transpose(1, 2, 0)
            # TODO: 多通道融合方法
            # imgtif = 0.114 * imgtif[:, :, 0] + 0.587 * imgtif[:, :, 1] + 0.299 * imgtif[:, :, 2]
            # imgtif = imgtif[:, :, np.newaxis].repeat(3, axis=2)
        else:  # 单通道图像
            imgtif = strentch_img(imgtif, norm_mode, np.uint8)
            imgtif = imgtif[:, :, np.newaxis].repeat(3, axis=2)
    
    del dataset
    gc.collect()
    
    if return_info:
        return imgtif,EPSGCode,im_proj,im_geotrans
    return imgtif


def save_RSTif(im_data, im_proj, im_geotrans, save_path):
    """保存遥感栅格文件
    Args:
        im_data: 输入的图像矩阵
        im_proj: 地图投影信息
        im_geotrans: 仿射矩阵
        save_path: 保存文件路径
    Returns:
        None
    """
    if 'int8' in im_data.dtype.name:  # 判断栅格数据的数据类型
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:       # 判读数组维数
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape 

    driver = gdal.GetDriverByName("GTiff")    # 创建文件, 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(save_path, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)      # 写入仿射变换参数
    dataset.SetProjection(im_proj)            # 写入投影
    if im_bands == 1:                         # 写入数组数据
        dataset.GetRasterBand(1).WriteArray(im_data)  
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def resize_RSTif(file_path, save_path, scale=1.0):
    """等比例缩放遥感栅格矩阵
    Args:
        file_path: 输入的文件路径
        save_path: 保存文件路径
        scale: 缩放因子
    Returns:
        None
    """
    if scale <= 0.0:
        raise ValueError(f"[ERROR] The scale {scale} is too small!")
    dataset = gdal.Open(file_path)      # 打开文件
    if dataset == None:
        print(f"[WARNING] Can not open file: {file_path}!")
        return
    cols = int(dataset.RasterXSize*scale)   # 栅格矩阵的列数
    rows = int(dataset.RasterYSize*scale)   # 栅格矩阵的行数
    geotrans = list(dataset.GetGeoTransform())   # 仿射矩阵
    geotrans[1] = geotrans[1] / scale  # 像元宽度变为原来的scale倍
    geotrans[5] = geotrans[5] / scale  # 像元高度变为原来的scale倍

    im_bands = dataset.RasterCount     # 波段数
    im_dtype = dataset.GetRasterBand(1).DataType   # 数据类型
    target = dataset.GetDriver().Create(save_path, xsize=cols, ysize=rows, 
                                        bands=im_bands, eType=im_dtype)
    target.SetProjection(dataset.GetProjection())  # 设置投影坐标
    target.SetGeoTransform(geotrans)               # 设置地理变换参数
    for index in range(1, im_bands+1):
        # 读取波段数据
        data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols,
                                                        buf_ysize=rows,
                                                        resample_alg=gdalconst.GRIORA_Average)
        """
        'GRIORA_Average', 'GRIORA_Bilinear', 'GRIORA_Cubic',
        'GRIORA_CubicSpline', 'GRIORA_Gauss', 'GRIORA_Lanczos',
        'GRIORA_Mode', 'GRIORA_NearestNeighbour'
        """
        out_band = target.GetRasterBand(index)
        if dataset.GetRasterBand(index).GetNoDataValue():
            out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
        out_band.WriteArray(data)         # 写入数据到新影像中
        out_band.FlushCache()
        out_band.ComputeBandStats(False)  # 计算统计信息
    del dataset, target


def tif2png_single(file_path,png_root,overwrite=True,scale=1.0,ext='png',norm_mode=1):
    """单张Tif图转rgb图像"""
    file_name = os.path.basename(file_path)
    if not overwrite and os.path.splitext(file_name)[0]+'.'+ext in os.listdir(png_root):
        return
    file_ext = os.path.splitext(file_name)[-1]
    JPGext = ['png','jpg','jpeg','PNG','JPG','JPEG']
    TIFext = ['tif','tiff','TIF','TIFF']
    # open image
    if file_name.endswith(tuple(TIFext)):
        img = read_RSTif(file_path, norm=True, norm_mode=norm_mode)
    elif file_name.endswith(tuple(JPGext)):
        img = cv2.imread(file_path)
    else:
        print(f'[WARNING] File Ext not in {TIFext} or {JPGext}: {file_path}.')
        return
    
    resizeimg = resize_image(img,scale)
    cv2.imwrite(os.path.join(png_root,file_name.replace(file_ext,f'.{ext}')),resizeimg)


def tif2png(tif_root,png_root,overwrite=True,scale=1,ext='png',norm_mode=1):
    """Tif图转rgb图像"""
    if not os.path.exists(png_root):
        os.makedirs(png_root)
    for file in tqdm.tqdm(os.listdir(tif_root), desc=f'tif2{ext}'):
        file_path = os.path.join(tif_root, file)
        tif2png_single(file_path, png_root, overwrite=overwrite, 
                       scale=scale, ext=ext, norm_mode=norm_mode)


def tif2png_multi_process(tif_root, png_root, overwrite=True, scale=1, process=8, ext='png', norm_mode=1):
    """多线程Tif图转rgb图像"""
    if not os.path.exists(png_root):
        os.makedirs(png_root)
    my_pool = Pool(process)
    filepath_list = [os.path.join(tif_root, filename) for filename in os.listdir(tif_root)]
    pbar = tqdm.tqdm(total=len(filepath_list))
    worker = partial(tif2png_single, 
                    png_root=png_root,
                    scale=scale, 
                    ext=ext, 
                    overwrite=overwrite,
                    norm_mode=norm_mode)
    pbar_iter = math.ceil(len(filepath_list)/50)
    for i in range(pbar_iter):
        my_pool.map(worker, filepath_list[i*50: (i+1)*50])
        pbar.update(50)


def get_position(width, height, subsize=10000, slide=8000):
    ps = []
    left, up = 0, 0
    while (left < width):
        if (left + subsize >= width):
            left = max(width - subsize, 0)
        up = 0
        while (up < height):
            if (up + subsize >= height):
                up = max(height - subsize, 0)
            ps.append((left, up))
            if (up + subsize >= height):
                break
            else:
                up = up + slide
        if (left + subsize >= width):
            break
        else:
            left = left + slide
    return ps


def split_tif_single(file_path, subsize, slide, save_root, overwrite=True, scale=1.0):
    dataset = gdal.Open(file_path)           # 打开文件
    if dataset == None:
        print(f"[WARNING] Can not open file: {file_path}!")
        return 
    im_w = dataset.RasterXSize               # 栅格矩阵的列数
    im_h = dataset.RasterYSize               # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj= dataset.GetProjection()         # 地图投影信息
    cut_locations = get_position(im_w, im_h, subsize, slide)
    for location in cut_locations:
        left, up = location
        imgtif = dataset.ReadAsArray(left, up, subsize, subsize)
        split_file = os.path.splitext(os.path.basename(file_path))[0] + \
                     f'__{scale}__{left}___{up}{os.path.splitext(os.path.basename(file_path))[-1]}'
        if not overwrite and split_file in os.listdir(save_root):
            continue
        save_RSTif(imgtif, im_proj, im_geotrans, os.path.join(save_root,split_file))


def split_tif(tif_root,save_root,subsize,slide,scale=1.0,overwrite=True):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for file in tqdm.tqdm(os.listdir(tif_root), desc=f'split_tif'):
        filepath = os.path.join(tif_root,file)
        split_tif_single(filepath,subsize,slide,save_root,overwrite=overwrite,scale=scale)


def split_tif_multi_process(tif_root, subsize, slide, save_root, overwrite=True, scale=1.0, process=8):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    my_pool = Pool(process)
    filepath_list = [os.path.join(tif_root, filename) for filename in os.listdir(tif_root)]
    pbar = tqdm.tqdm(total=len(filepath_list))
    worker = partial(split_tif_single,
                    subsize=subsize, 
                    slide=slide,
                    save_root=save_root,
                    scale=scale, 
                    overwrite=overwrite)
    pbar_iter = math.ceil(len(filepath_list)/50)
    for i in range(pbar_iter):
        my_pool.map(worker, filepath_list[i*50: (i+1)*50])
        pbar.update(50)


def split_tif2png_single(filepath,subsize,slide,png_root,overwrite=True,scale=1.0,norm_mode=1,ext='png'):
    imgtif = read_RSTif(filepath, norm=True, norm_mode=norm_mode)
    im_h, im_w = imgtif.shape[:2]
    cut_locations = get_position(im_w, im_h, subsize, slide)
    for location in cut_locations:
        left, up = location
        imgarr = imgtif[up:up+subsize, left:left+subsize, :]
        imgarr = resize_image(imgarr,scale)
        split_file = os.path.splitext(os.path.basename(filepath))[0] + f'__{scale}__{left}___{up}.{ext}'
        if not overwrite and split_file in os.listdir(png_root):
            continue
        cv2.imwrite(os.path.join(png_root,split_file),imgarr)


def split_tif2png(tif_root,png_root,subsize,slide,scale=1.0,overwrite=True,norm_mode=1,ext='png'):
    if not os.path.exists(png_root):
        os.makedirs(png_root)
    for file in tqdm.tqdm(os.listdir(tif_root), desc=f'split_tif2{ext}'):
        filepath = os.path.join(tif_root,file)
        split_tif2png_single(filepath,subsize,slide,png_root,overwrite=overwrite,scale=scale,norm_mode=norm_mode,ext=ext)


def split_tif2png_multi_process(tif_root,png_root,subsize,slide,scale=1.0,overwrite=True,norm_mode=1,ext='png',process=8):
    if not os.path.exists(png_root):
        os.makedirs(png_root)
    my_pool = Pool(process)
    filepath_list = [os.path.join(tif_root, filename) for filename in os.listdir(tif_root)]
    pbar = tqdm.tqdm(total=len(filepath_list))
    worker = partial(split_tif2png_single, 
                    png_root=png_root,
                    subsize=subsize, 
                    slide=slide, 
                    scale=scale, 
                    ext=ext, 
                    overwrite=overwrite,
                    norm_mode=norm_mode)
    pbar_iter = math.ceil(len(filepath_list)/50)
    for i in range(pbar_iter):
        my_pool.map(worker, filepath_list[i*50:(i+1)*50])
        pbar.update(50)


def img2geo(geo_trans,x,y):
    """图像坐标系转地理坐标系
    Args:
        geo_trans: 仿射矩阵六元组
        x: 图像x坐标
        y: 图像y坐标
    Returns:
        gx, gy: 地理坐标系
    """
    gx = geo_trans[0] + x * geo_trans[1] + y * geo_trans[2]
    gy = geo_trans[3] + x * geo_trans[4] + y * geo_trans[5]
    return gx,gy


def getSRSPair(im_proj):
    """获得给定数据的投影参考系和地理参考系
    Args:
        im_proj: GDAL投影信息
    Returns:
        prosrs, geosrs: 投影参考系和地理参考系
    """
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(im_proj)
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2lonlat(im_proj, gx, gy):
    """将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    Args:
        im_proj: GDAL投影信息
        gx: 投影坐标x
        gy: 投影坐标y
    Returns:
        投影坐标(gx, gy)对应的经纬度坐标(lon, lat)
    """
    prosrs, geosrs = getSRSPair(im_proj)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(gx, gy)
    return coords[:2]


def geo2imagexy(geo_trans, gx, gy):
    """根据GDAL的仿射矩阵将给定的投影/地理坐标转为图像坐标
    Args:
        geo_trans: 仿射矩阵
        gx: 投影或地理坐标x
        gy: 投影或地理坐标y
    Returns:
        投影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    """
    a = np.array([[geo_trans[1], geo_trans[2]], [geo_trans[4], geo_trans[5]]])
    b = np.array([gx - geo_trans[0], gy - geo_trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


def lonlat2geo(lon, lat, source_epsg=4326, target_epsg=32652):
    """根据经纬度转为地理/投影坐标
    Args:
        gx: 投影或地理坐标x
        gy: 投影或地理坐标y
    Returns:
        投影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    """
    source = osr.SpatialReference()
    source.ImportFromEPSG(source_epsg)
    if source_epsg == 4326:
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(target_epsg)
    if target_epsg == 4326:
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    return transform.TransformPoint(lon,lat)[:2]


if __name__ == "__main__":
    print("""[Utils for tif files]
          It contains various functions, such as:
          - strentch_img
          - read_RSTif
          - save_RSTif
          - resize_RSTif
          - tif2png_single, tif2png, tif2png_multi_process
          - get_position
          - split_tif_single, split_tif, split_tif_multi_process
          - split_tif2png_single, split_tif2png, split_tif2png_multi_process
          - ...
          """)