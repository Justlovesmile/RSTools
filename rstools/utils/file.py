import os
import shutil as sh
from multiprocessing import Pool


def custom_basename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def get_file_from_this_dir(dir, ext = None):
    allfiles = []
    needExtFilter = (ext != None)
    if type(ext) == str:
        ext = [ext]
    for eid, e in enumerate(ext):
        ext[eid] = e if e.startswith('.') else f".{e}"
    for root, _, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root,file)
            extension = os.path.splitext(os.path.basename(filepath))[-1]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def single_copy(src_dst_tuple):
    sh.copyfile(*src_dst_tuple)

def filecopy(srcpath, dstpath, num_process=64):
    pool = Pool(num_process)
    filelist = get_file_from_this_dir(srcpath)
    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)
    pool.map(single_copy, name_pairs)


def singel_move(src_dst_tuple):
    sh.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=64):
    pool = Pool(num_process)
    filelist = get_file_from_this_dir(srcpath)
    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)
    pool.map(filemove, name_pairs)


def getnamelist(srcpath, dstfile, ext=None, keep_ext=False):
    filepath_list = get_file_from_this_dir(srcpath, ext)
    with open(dstfile, 'w') as f_out:
        for file in filepath_list:
            if not keep_ext:
                file = os.path.splitext(os.path.basename(file))[0]
            f_out.write(file + '\n')


def getnamelist_from_two_dirs(srcpath1, srcpath2, dstfile, ext1=None, ext2=None):
    filepath_list1 = get_file_from_this_dir(srcpath1, ext1)
    filepath_list2 = get_file_from_this_dir(srcpath2, ext2)
    fileList1 = [os.path.splitext(os.path.basename(filepath))[0] for filepath in filepath_list1]
    fileList2 = [os.path.splitext(os.path.basename(filepath))[0] for filepath in filepath_list2]
    same_name = set(fileList1)&set(fileList2)
    with open(dstfile, 'w') as f_out:
        for name in same_name:
            f_out.write(name+'\n')

if __name__ == "__main__":
    import sys
    datadir,ext = sys.argv[1], sys.argv[2:]
    files = get_file_from_this_dir(datadir,list(ext))
    print(len(files),files)
    