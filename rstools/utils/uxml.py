import xml.etree.ElementTree as ET

def noneType(value):
    return value

def getChildByName(parentNode,name):
    return parentNode.find(name)

def getAllChildByName(parentNode,name):
    return parentNode.findall(name)

def getChildValueByName(parentNode,name,Type):
    return Type(parentNode.find(name).text)

def getChildsValueByNameList(parentNode,nameList,Type=noneType):
    childs_value = []
    if len(set(nameList)) == len(nameList):
        for name in nameList:
            childs_value.append(getChildValueByName(parentNode,name,Type))
        return childs_value
    else:
        for name in set(nameList):
            childs = getAllChildByName(parentNode,name)
            for ch in childs:
                childs_value.append(Type(ch.text))
        return childs_value


def parseDict(root,d):
    res = {}
    for key, value in d.items():
        if key not in res.keys():
            res[key]=[]
        if type(value)==dict:
            childs = getAllChildByName(root,key)
            for ch in childs:
                rv = parseDict(ch,value)
                res[key].append(rv)
        elif type(value)==list:
            assert len(value)>=1
            Type=value[0]
            if len(value)==1:
                res[key] = getChildValueByName(root,key,Type)
            else:
                childs = getAllChildByName(root,key)
                for ch in childs:
                    if len(set(value[1:])) == len(value[1:]):
                        res[key] = getChildsValueByNameList(ch,value[1:],Type)
                    else:
                        res[key].append(getChildsValueByNameList(ch,value[1:],Type))
        
    return res

def parse_xml(xmlPath,nodeDict=None):
    """
    nodeDict:
        # 1. XML Example
        <?xml version="1.0" encoding="UTF-8"?>
        <annotation>
            <filename>00001.jpg</filename>
            <size>
                <width>800</width>
                <height>800</height>
                <depth>3</depth>
            </size>
            <object>
                <name>car</name>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>133</xmin>
                    <ymin>237</ymin>
                    <xmax>684</xmax>
                    <ymax>237</ymax>
                </bndbox>
                <points><!--Extreme Situation: the same name of label-->
                    <item>
                        <item>100</item>
                        <item>200</item>
                    </item>
                    <item>
                        <item>300</item>
                        <item>400</item>
                    </item>
                    <item>
                        <item>500</item>
                        <item>600</item>
                    </item>
                    <item>
                        <item>700</item>
                        <item>800</item>
                    </item>
                </points>
            </object>
        </annotation>
        # 2. Dict Example
        nodeDict = {
            'annotation': {
                'filename': [str],
                'size': [int,'width','height','depth'],
                'object': {
                    'name': [str],
                    'difficult': [int],
                    'bndbox': [float,'xmin','ymin','xmax','ymax'],
                    'points': {
                        'item': [float, 'item', 'item'],
                    },
                },
            },
        }
        # 3. Return Example
        {
            'annotation': [{
                'filename': '00001.jpg', 
                'size': [800, 800, 3], 
                'object': [{
                    'name': 'car', 
                    'difficult': 0, 
                    'bndbox': [133.0, 237.0, 684.0, 237.0], 
                    'points': [{
                        'item': [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0], [700.0, 800.0]]
                    }]
                }]
            }]
        }
    """
    assert nodeDict and type(nodeDict)==dict, f"[ERROR] Unsupported nodeDict"
    Tree = ET.parse(xmlPath)
    Root = Tree.getroot()
    
    res = {}
    for key,value in nodeDict.items():
        res[key] = [parseDict(Root,value)]
        
    return res