import os
import json
import tqdm
import numpy as np
from osgeo import gdal, osr, ogr

from rstools.utils.file import get_file_from_this_dir

def rslabel2dota(jsondir,txtdir):
    filepaths = get_file_from_this_dir(jsondir,ext='json')
    for filepath in tqdm.tqdm(filepaths):
        txtpath = os.path.join(txtdir,os.path.basename(filepath).replace('.json','.txt'))  
        with open(filepath,'r', encoding='utf-8') as f:
            data = json.load(f)
        geotrans = data.get('geoTrans','geo not exist')           
        with open(txtpath, 'w', encoding='utf-8') as fp:
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
                # TODO: process label
                # END
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

                fp.write('%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s %d\n'%(pbox_geo[0],pbox_geo[1],pbox_geo[2],pbox_geo[3],pbox_geo[4],pbox_geo[5],pbox_geo[6],pbox_geo[7],label,difficult))

def rslabel2shp(datadir):
    jsonpaths = get_file_from_this_dir(datadir,ext='json')
    for jsonpath in jsonpaths:
        tifpath = jsonpath.replace('.json','.tif')
        tiffpath = jsonpath.replace('.json','.tiff')
        shppath = jsonpath.replace('.json','.shp')
        if os.path.exists(tiffpath):
            dataset = gdal.Open(tiffpath)
        elif os.path.exists(tifpath):
            dataset = gdal.Open(tifpath)
        else:
            continue

        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        ogr.RegisterAll()
        srs = osr.SpatialReference()
        im_w = dataset.RasterXSize             # 栅格数据的列数
        im_h = dataset.RasterYSize             # 栅格数据的行数
        im_geotrans=dataset.GetGeoTransform()  # 栅格数据的仿射矩阵
        im_proj = dataset.GetProjection()      # 栅格数据的投影信息
        srs.ImportFromWkt(im_proj)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(shppath)
        layer = data_source.CreateLayer(shppath, srs, ogr.wkbMultiPolygon)
        field_name = ogr.FieldDefn("label", ogr.OFTString)
        field_name.SetWidth(250)
        layer.CreateField(field_name)
        feature = ogr.Feature(layer.GetLayerDefn())

        with open(jsonpath,'r') as f:
            content = json.load(f)
        shapes = content['shapes']
        
        for i in range(len(shapes)):
            feature.SetField("label", shapes[i]["label"])
            if len(shapes[i]["points"])==4:
                x1,y1=shapes[i]["points"][0]
                x2, y2 = shapes[i]["points"][1]
                x3, y3 = shapes[i]["points"][2]
                x4, y4 = shapes[i]["points"][3]
                polygon="POLYGON(({0} {1}, {2} {3}, {4} {5}, {6} {7}, {8} {9}))".format(x1,y1,x2,y2,x3,y3,x4,y4,x1,y1)
                geomRectangle = ogr.CreateGeometryFromWkt(polygon)
                feature.SetGeometry(geomRectangle)
                layer.CreateFeature(feature)
            if len(shapes[i]["points"])==8:
                x1,y1=shapes[i]["points"][0]
                x2, y2 = shapes[i]["points"][2]
                x3, y3 = shapes[i]["points"][4]
                x4, y4 = shapes[i]["points"][6]
                polygon="POLYGON(({0} {1}, {2} {3}, {4} {5}, {6} {7}, {8} {9}))".format(x1,y1,x2,y2,x3,y3,x4,y4,x1,y1)
                geomRectangle = ogr.CreateGeometryFromWkt(polygon)
                feature.SetGeometry(geomRectangle)
                layer.CreateFeature(feature)
        feature = None
        data_source = None


def shp2rslabel(datadir):
    def readShp(filename):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8") #GBK or UTF-8
        ogr.RegisterAll()
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(filename, 0)
        if ds == None:
            print(f"[ERROR] Open shp file failed! {filename}")
            return 
        oLayer = ds.GetLayerByIndex(0)
        if oLayer == None:
            print(f"[ERROR] Get layer by index failed! {filename}")
            return 
        oLayer.ResetReading()
        num = oLayer.GetFeatureCount(0)
        result_list = []
        for i in range(0, num):
            ofeature = oLayer.GetFeature(i)
            label = ofeature.GetFieldAsString("label")
            geom = str(ofeature.GetGeometryRef())
            result_list.append([label,geom])
        ds.Destroy()
        del ds
        return result_list, num

    shppaths = get_file_from_this_dir(datadir,ext='shp')
    for shppath in shppaths:
        result_list, num = readShp(shppath)
        tifpath = shppath.replace('.shp','.tif')
        tiffpath = shppath.replace('.shp','.tiff')
        jsonpath = shppath.replace('.shp','.json')
        if os.path.exists(tiffpath):
            imgpath = tiffpath
        elif os.path.exists(tifpath):
            imgpath = tifpath
        else:
            continue

        dataset = gdal.Open(imgpath)
        im_w = dataset.RasterXSize             # 栅格数据的列数
        im_h = dataset.RasterYSize             # 栅格数据的行数
        im_geotrans=dataset.GetGeoTransform()  # 栅格数据的仿射矩阵
        result_list, num = readShp(shppath)
        shapes = []
        if num>0:
            for i in result_list:
                #print(i)
                label=i[0]
                geom=i[1]
                geom=geom[10:-2]
                point1=[float(geom.split(",")[0].split(" ")[0]),float(geom.split(",")[0].split(" ")[1])]
                point2=[float(geom.split(",")[1].split(" ")[0]),float(geom.split(",")[1].split(" ")[1])]
                point3=[float(geom.split(",")[2].split(" ")[0]),float(geom.split(",")[2].split(" ")[1])]
                point4=[float(geom.split(",")[3].split(" ")[0]),float(geom.split(",")[3].split(" ")[1])]
                shape={
                    "label": label,
                    "line_color": [0,0,0,255],
                    "fill_color": [0,0,0,255],
                    "points": [point1,point2,point3,point4],
                    "probability": 10.0,
                    "shape_type": "slantRectangle"
                }
                shapes.append(shape)
            dict={
                "version":1,
                "flags":{},
                "shapes":shapes,
                "lineColor": [0,255,0,128],
                "fillColor": [255,0,0,128],
                "imagePath": imgpath,
                "imageData": None,
                "imageHeight": im_h,
                "imageWidth": im_w,
                "geoTrans": im_geotrans
            }
            with open(jsonpath, 'w', encoding='utf-8') as f:
                json.dump(dict, f, ensure_ascii=False,indent=2)
        else:
            print("[WARNING] shp file is empty with annotation! {shppath}")

if __name__ == "__main__":
    import sys
    datadir = sys.argv[1]
    shp2rslabel(datadir)