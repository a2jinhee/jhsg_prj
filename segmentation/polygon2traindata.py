"""
    polygon2traindata.py

    Authored Sept. 23, 2021 by Youngtak Cho @ GazziLabs Co., Ltd.
"""

import argparse
import json
from json.decoder import JSONDecodeError
import numpy as np
import os
from PIL import Image, ImageDraw, ImageOps
import random
import textwrap

#from numpy.core.fromnumeric import reshape
from print_log import *

CONST_VER_MAJOR = 1
CONST_VER_MINOR1 = 2
CONST_VER_MINOR2 = 3
CONST_VER_DATE = 220127

# file error id
CONST_ERROR_0001 = "ERR0001"  # ERR0001 : json values err
CONST_ERROR_0002 = "ERR0002"  # ERR0002 : exceed boundary
CONST_ERROR_0003 = "ERR0003"  # ERR0003 : nothing to do, just skip
CONST_ERROR_0004 = "ERR0004"  # ERR0004 : invalid info argument
CONST_ERROR_0005 = "ERR0005"  # ERR0005 : shape and location mismatched
CONST_ERROR_0006 = "ERR0006"  # ERR0006 : no polygon or bounding box items in shape element
CONST_ERROR_0007 = "ERR0007"  # ERR0007 : invlaid coordinates
CONST_ERROR_0008 = "ERR0008"  # ERR0008 : no bounding boxes
CONST_ERROR_0009 = "ERR0009"  # ERR0009 : bounding update err
CONST_ERROR_0010 = "ERR0010"  # ERR0010 : boundingbox crop err
CONST_ERROR_0011 = "ERR0011"  # ERR0011 : masking image unexpected error
CONST_ERROR_0012 = "ERR0012"  # ERR0012 : size or orientation mismatched between input image and its metadata
CONST_ERROR_0013 = "ERR0013"  # ERR0013 : Labeling area not large enough

# file format for annotated metadata
CONST_METADATA_EXT = ".json"

CONST_FOLDER_INPUT = "_input"
CONST_FOLDER_MASK = "_mask"
CONST_FOLDER_BBOX = "_bbox"
CONST_FOLDER_JSON = "_json"
CONST_FOLDER_PREVIEW = "_preview"
CONST_FOLDER_LOG = "logs"

CONST_FILELIST_PROCESSED = "_processed.lst"
CONST_FILELIST_ACCEPTED = "_accepted.lst"
CONST_FILELIST_IGNORED = "_ignored.lst"

# 'p': with processed.lst
# 'i': with ignored.lst
# 'p-i': with processed.lst - ignored.lst
CONST_REWORK_MODE = ( "a", "i", "p", "p-i" )

CONST_MIN_AREARATIO = 0.0   # accept all
CONST_MAX_AREARATIO = 0.99  # exclude if labeled area under 99 percent of image

CONST_IMG_QUALITY = 95
CONST_IMG_SUBSAMPLE = 0

MSG_BUFFER = []


class Config:
    def __init__(self):
        # only for test
        self.TEST_DATA_PATH = "data" + os.sep + "samples"

        # dir path for input data
        self.SET_DATA_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + os.sep + ".." + os.sep + self.TEST_DATA_PATH)

        # dir path for output data
        self.SET_OUTPUT_PATH = self.SET_DATA_PATH

        # hint for metadata structure
        self.SET_MEATDATA_STRUCT = {
            'filename': ('metaData', 'Raw data ID'),
            'resolution': ('metaData', 'resolution'),
            'shape': ('labelingInfo', 'polygon', 'type'),
            'color': ('labelingInfo', 'polygon', 'color'),
            'location': ('labelingInfo', 'polygon', 'location'),
            'label': ('labelingInfo', 'polygon', 'label'),
            'box': ('labelingInfo', 'box', 'type'),
            'boxlocation': ('labelingInfo', 'box', 'location'),
            }

        # file name decoration for output mask image (default: 'mask')
        self.SET_FILENAME_MASKDECO = "mask"

        # file name decoration for output mask image (default: 'mask')
        self.SET_FILENAME_BBOXDECO = "bbox"

        # if True, add string (SET_FILENAME_MASKDECO + "_") as prefix of each file name
        # otherwise, add string ("_" + SET_FILENAME_MASKDECO) as postfix of each file name
        # SET_FILENAME_BBOXDECO is same
        # (default: not set)
        self.SET_DECO_FOR_PREFIX = False

        # find files in sub-folders recursively (default: not set)
        self.SET_FIND_RECURSIVELY = False

        # set 'creates and saves polygon label-related mask images' mode
        self.SET_CREATE_MASKIMG = False

        # set 'crop mask region with SET_CROP_MARGIN' mode
        self.SET_CROP_MASK = False

        # set 'crop margin' in pixel-wide (default: 10 for no marginal left, right, top, bottom)
        self.SET_CROP_MARGINE = 10

        # set 'creates and updates bounding box info' mode
        self.SET_UPDATE_BBOXINFO = False

        # set 'crop bbox-relatived images' mode
        self.SET_CROP_BBOX = False

        # set 'aggregates classes' mode
        self.SET_AGGREGATE_CLASSES = False

        # set 'copy original files and create previews' mode
        self.SET_CREATE_EXTRAS = True

        # set 'resume from the last file of the previous same job ID' mode (Mutually excluded from SET_REWORK)
        self.SET_RESUME = False

        # set 'rework previous job with same job ID' mode (Mutually excluded from SET_RESUME)
        self.SET_REWORK = False

        # set 'rework mode' for option --rework
        self.SET_REWORK_MODE = 1

        # set 'minimum area ratio' as threshold for labeled area (px) = (image width x height) * SET_MIN_AREA_RATIO
        self.SET_MIN_AREA_RATIO = CONST_MIN_AREARATIO

        # set 'auto rotate' mode
        self.SET_ROTATE_WITH_EXIF = False


def logout(msg, stdout = False, force_flush = False):
    print_log(level = 'i', msg = msg, tag = "Polygon2TrainData", on_screen_display = (stdout or is_debug()), force_flush = force_flush)


def __parseHint(path):
    temp_hint_dict = {}

    if os.path.exists(path) == False or os.path.isfile(path) == False:
        logout('the hint file for metadata structure not exist! check --hint option:\n\t{0}'.format(path))
        return None

    return temp_hint_dict


def __getArguments():
    parser = argparse.ArgumentParser(prog="polygon2traindata",
        description=textwrap.dedent(f'''
            ================================================
             Data Preparation Utilities - Polygon2TrainData
             ver.{CONST_VER_MAJOR}.{CONST_VER_MINOR1}.{CONST_VER_MINOR2}.{CONST_VER_DATE}
            ================================================'''),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            --------------------------------------
             Authored by Youngtak Cho
             Sept. 23, 2021 @ GazziLabs Co., Ltd.
            --------------------------------------

            Composite options can be considered to handle complex tasks:
            for example, updating metadata for bounding boxes and creating cropped images
            while simultaneously generating mask images.
            You may also need to collect the processed files into each class folder.

            If so, try this:
            ---------------------
            python polygon2traindata.py --input "YOUR_SOURCE_PATH" --output "YOUR_OUTPUT_PATH" --updatebbox --cropbbox --maskimg --recursive --aggregate
            '''))
    parser.add_argument('-i', '--input', metavar='PATH', required=False, default=os.path.normpath(os.getcwd()),
                        help='dir or file path to labeled data as input. if not set, it will find from current directory.')
    parser.add_argument('-o', '--output', metavar='DIR_PATH', required=True,
                        help='if set, the converted files are output to the specified path. if not, the files are output to the specified --input path. (default: not set)')
    parser.add_argument('--hint', metavar='JSON_PATH', required=False, default="",
                        help='set hint file for metadata structure info (only JSON format is supported)')
    parser.add_argument('--maskdeco', metavar='STRING', required=False, default='mask',
                        help='file name decoration for mask (default: \'mask\')')
    parser.add_argument('--bboxdeco', metavar='STRING', required=False, default='bbox',
                        help='file name decoration for bbox (default: \'bbox\')')
    parser.add_argument('--asprefix', required=False, action='store_true', default=False,
                        help='if set, the string specified with the --deco option is used as a prefix for the output file name. otherwise, it used as a suffix. (default: not set)')
    parser.add_argument('--recursive', required=False, action='store_true', default=False,
                        help='if set, find data files in sub folders (default: not set)')

    parser.add_argument('--maskimg', required=False, action='store_true', default=False,
                        help='if set, creates and saves polygon label-related mask images. (default: not set)')
    parser.add_argument('--cropmask', required=False, action='store_true', default=False,
                        help='if set, crop mask region with SET_CROP_MARGIN. (default: not set)')
    parser.add_argument('--cropmargin', required=False, default=10, type=int,
                        help='if set, crop margin in pixel-wide (default: 10 for no marginal left, right, top, bottom)')

    parser.add_argument('--updatebbox', required=False, action='store_true', default=False,
                        help='if set, creates and/or updates bounding box info in the JSON metafile. (default: not set)')
    parser.add_argument('--cropbbox', required=False, action='store_true', default=False,
                        help='if set, crops and saves bbox-related images. (default: not set)')

    parser.add_argument('--arearatio', metavar="INT", required=False, default=int(CONST_MIN_AREARATIO*100), type=int, choices=range(0, 100),
                        help=f'if set, only images that exceed the minimum size ratio to the total area are selected. available range: [{int(CONST_MIN_AREARATIO*100)} .. {int(CONST_MAX_AREARATIO*100)}] (default: {int(CONST_MIN_AREARATIO*100)})')
    parser.add_argument('--autorotate', required=False, action='store_true', default=False,
                        help='if set, rotate image based on Exif orientation info automatically. (default: not set)')

    parser.add_argument('--aggregate', required=False, action='store_true', default=False,
                        help='if set, each class folder collects data of the same class using the last folder name as the class name. i.e) if path = "c:/data/in/user/folder", class name is \'folder\' (default: not set)')
    parser.add_argument('--noextras', required=False, action='store_true', default=False,
                        help='if set, copy original JPG and JSON files and create extra files under your --output path. (default: not set)')
                        
    parser.add_argument('--jobid', metavar='STRING', required=False, default="myjob",
                        help='set job ID for current convert task to enable --resume (default: \'myjob\')')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', required=False, action='store_true', default=False,
                        help='If set, the conversion process will start on the file last processed by the same previous job ID. (default: not set)')
    group.add_argument('--rework', required=False, action='store_true', default=False,
                        help='If set, the conversion process will restart based on ignored.lst or processed.lst with the same previous job ID. you must be set option --reworkmode. (default: not set)')
    parser.add_argument('--reworkmode', metavar='STRING', required=False, default='i', choices=['a', 'i', 'p', 'p-i'],
                        help='if --rework set, rework process running with list of (\'a\')ccepted, (\'p\')rocessed or (\'i\')gnored. You can also run it with a \'Processed\' list with \'Ignored\' excluded. (\'p-i\'). (default: \'i\')')

    parser.add_argument('--debug', required=False, action='store_true', default=False,
                        help='if set, running with debug mode for development (default: not set)')
                        
    args = parser.parse_args()

    return args


import pprint
def __parseArguments(args):
    config = Config()

    if args.output != "":
        config.SET_OUTPUT_PATH = os.path.abspath(args.output)
    else:
        config.SET_OUTPUT_PATH = os.path.abspath(config.SET_DATA_PATH)

    set_debug(args.debug)
    set_log_path(config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG, args.jobid)

    logout('input arguments:\n\t{0}'.format(vars(args)))

    if is_debug() == False:
        if args.input != "":
            config.SET_DATA_PATH = args.input
        else:
            logout("ERROR: input path not available! check --input option:\n\t{0}".format(args.input))
            exit(0)

    if args.hint != "":
        temp_hint_dict = __parseHint(args.hint)
        if temp_hint_dict is not None:
            config.SET_MEATDATA_STRUCT = temp_hint_dict

    if args.maskdeco != "":
        config.SET_FILENAME_MASKDECO = args.maskdeco

    if args.bboxdeco != "":
        config.SET_FILENAME_BBOXDECO = args.bboxdeco

    config.SET_DECO_FOR_PREFIX = args.asprefix
    config.SET_FIND_RECURSIVELY = args.recursive

    config.SET_CREATE_MASKIMG = args.maskimg
    config.SET_CROP_MASK = args.cropmask
    config.SET_CROP_MARGINE = args.cropmargin
    if config.SET_CROP_MARGINE < 10:
        config.SET_CROP_MARGINE = 10
        logout("--cropmargin must be set 10 at least!")

    config.SET_UPDATE_BBOXINFO = args.updatebbox
    config.SET_CROP_BBOX = args.cropbbox

    config.SET_MIN_AREA_RATIO = float(args.arearatio/100.)
    config.SET_ROTATE_WITH_EXIF = args.autorotate

    config.SET_AGGREGATE_CLASSES = args.aggregate
    config.SET_CREATE_EXTRAS = (args.noextras == False)

    config.SET_RESUME = args.resume
    config.SET_REWORK = args.rework
    config.SET_REWORK_MODE = args.reworkmode

    logout('parsed arguments:\n\t{0:<16}{1}\n\t{2:<16}{3}\n\t{4:<16}{5}\n\t{6:<16}{7}\n\t{8:<16}{9}\n\t{10:<16}{11} (output file name: \'{12}.ext\')\n\t{13:<16}{14}\n\t{15:<16}{16}\n\t{17:<16}{18}\n\t{19:<16}{20}\n\t{21:<16}{22}\n\t{23:<16}{24}\n\t{25:<16}{26}\n\t{27:<16}{28}\n\t{29:<16}{30}\n\t{31:<16}{32}\n\t{33:<16}{34}\n\t{35:<16}{36}\n\t{37:<16}{38}\n\t{39:<16}{40}\n\t{41:<16}{42}'
        .format('--input', config.SET_DATA_PATH,
                '--output', config.SET_OUTPUT_PATH,
                '--hint', "{{\n\t\t\t    {0}\n\t\t        }}".format("\n\t\t\t    ".join("'{0}': {1}".format(key, value) for key, value in config.SET_MEATDATA_STRUCT.items())),
                '--maskdeco', config.SET_FILENAME_MASKDECO,
                '--bboxdeco', config.SET_FILENAME_BBOXDECO,
                '--asprefix', config.SET_DECO_FOR_PREFIX, '[filename]_' * (config.SET_DECO_FOR_PREFIX == False) + '[--maskdeco or --bboxdeco]' + '_[filename]' * config.SET_DECO_FOR_PREFIX,
                '--recursive', config.SET_FIND_RECURSIVELY,
                '--maskimg', config.SET_CREATE_MASKIMG,
                '--cropmask', config.SET_CROP_MASK,
                '--cropmargin', config.SET_CROP_MARGINE,
                '--updatebbox', config.SET_UPDATE_BBOXINFO,
                '--cropbbox', config.SET_CROP_BBOX,
                '--arearatio', config.SET_MIN_AREA_RATIO,
                '--autoratate', config.SET_ROTATE_WITH_EXIF,
                '--aggregate', config.SET_AGGREGATE_CLASSES,
                '--noextras', config.SET_CREATE_EXTRAS == False,
                '--resume', config.SET_RESUME,
                '--rework', config.SET_REWORK,
                '--reworkmode', config.SET_REWORK_MODE,
                '--jobid', getJobId(),
                '--debug', is_debug()), True)
    if is_debug():
        logout('WARNING! if you want to run with your data, remove --debug option')

    return config


import glob
import shutil
def __collectIgnoredSample(filepath, config):
    if os.path.isdir(filepath):
        return

    path, file = os.path.split(filepath)
    classname = os.path.basename(path)

    path_ignored = config.SET_OUTPUT_PATH + os.sep + "_ignored"
    if not os.path.exists(path_ignored):
        os.makedirs(path_ignored)

    path_class = path_ignored + os.sep + classname
    if not os.path.exists(path_class):
        os.makedirs(path_class)

    filename, ext = os.path.splitext(file)
    try:
        for file in glob.glob(path + os.sep + filename + ".*"):
            shutil.copy2(file, path_class)
    except Exception as e:
        logout("collectIgnoredSample failed:\n{0}".format(e))


ERROR_JSON = 'JSON'
ERROR_MASK = 'MASK'
ERROR_BBOX = 'BBOX'

BUFFER_PROCESSED = []
BUFFER_ACCEPTD = []
BUFFER_IGNORED = []

CONST_FLUSH_COUNT = 100

def __updateProcessedList(entrypath, config:Config, flush = False):
    global CONST_FILELIST_PROCESSED
    global BUFFER_PROCESSED
    global CONST_FLUSH_COUNT

    if len(entrypath) > 0:
        BUFFER_PROCESSED.append(entrypath)
    if flush == False:
        if len(BUFFER_PROCESSED) < CONST_FLUSH_COUNT:
            return
    else:
        if len(BUFFER_PROCESSED) == 0:
            return

    rework = "_rework" if config.SET_REWORK else ""
    listfile = config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG + os.sep + f"{getJobId()}{rework}{CONST_FILELIST_PROCESSED}"
    with open(listfile, 'a', encoding='utf-8') as flist:
        temp = "\n".join(BUFFER_PROCESSED)
        flist.write(f"{temp}\n")
        flist.flush()
        BUFFER_PROCESSED.clear()


def __updateAcceptedList(entrypath, config:Config, flush = False):
    global CONST_FILELIST_ACCEPTED
    global BUFFER_ACCEPTD
    global CONST_FLUSH_COUNT

    if len(entrypath) > 0:
        BUFFER_ACCEPTD.append(entrypath)
    if flush == False:
        if len(BUFFER_ACCEPTD) < CONST_FLUSH_COUNT:
            return
    else:
        if len(BUFFER_ACCEPTD) == 0:
            return

    rework = "_rework" if config.SET_REWORK else ""
    listfile = config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG + os.sep + f"{getJobId()}{rework}{CONST_FILELIST_ACCEPTED}"
    with open(listfile, 'a', encoding='utf-8') as flist:
        temp = "\n".join(BUFFER_ACCEPTD)
        flist.write(f"{temp}\n")
        flist.flush()
        BUFFER_ACCEPTD.clear()


def __updateIgnoredList(filepath, ecode, errid, config:Config, flush = False):
    global CONST_FILELIST_IGNORED
    global BUFFER_IGNORED
    global CONST_FLUSH_COUNT

    if os.path.isdir(filepath):
        return

    path, file = os.path.split(filepath)
    filename, ext = os.path.splitext(file)

    if len(filepath) > 0:
        for file in glob.glob(path + os.sep + filename + ".*"):
            BUFFER_IGNORED.append(f"{file}|{ecode}|{errid}\n")
    if flush == False:
        if len(BUFFER_IGNORED) < CONST_FLUSH_COUNT:
            return
    else:
        if len(BUFFER_IGNORED) == 0:
            return

    rework = "_rework" if config.SET_REWORK else ""
    listfile = config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG + os.sep + f"{getJobId()}{rework}{CONST_FILELIST_IGNORED}"
    with open(listfile, 'a', encoding='utf-8') as flist:
        temp = "\n".join(BUFFER_IGNORED)
        flist.write(f"{temp}\n")
        flist.flush()
        BUFFER_IGNORED.clear()


def __getPrevProcessedList(config:Config):
    global CONST_FILELIST_PROCESSED

    prevProcessedList = []
    try:
        listfile = config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG + os.sep + f"{getJobId()}{CONST_FILELIST_PROCESSED}"
        with open(listfile, 'r', encoding='utf-8') as flist:
            prevProcessedList = flist.read().splitlines()
    except Exception as e:
        logout("read previous processed list failed:\n{0}".format(e))
    
    return prevProcessedList


def __getPrevAcceptedList(config:Config):
    global CONST_FILELIST_ACCEPTED

    prevProcessedList = []
    try:
        listfile = config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG + os.sep + f"{getJobId()}{CONST_FILELIST_ACCEPTED}"
        with open(listfile, 'r', encoding='utf-8') as flist:
            prevProcessedList = flist.read().splitlines()
    except Exception as e:
        logout("read previous accepted list failed:\n{0}".format(e))
    
    return prevProcessedList


def __getPrevIgnoredList(config:Config):
    global CONST_FILELIST_IGNORED

    prevIgnoredList = []
    try:
        listfile = config.SET_OUTPUT_PATH + os.sep + CONST_FOLDER_LOG + os.sep + f"{getJobId()}{CONST_FILELIST_IGNORED}"
        with open(listfile, 'r', encoding='utf-8') as flist:
            prevIgnoredList = flist.read().splitlines()
    except Exception as e:
        logout("read previous ignored list failed:\n{0}".format(e))
    
    return prevIgnoredList


def __getTerminalItems(tag, doc, keys):
    for key in keys:
        # for 'polygon' and 'box'
        if key not in doc: return None
        doc = doc[key]

    result = []
    if tag == "location":   # for polygon
        # at least, we need 2 coordinates - (x, y) pairs
        if len(doc[0].values()) < 4 or len(doc[0].values()) % 2 != 0:
            result = None
        else:
            '''
            # ERROR) Oops! there are arbitrary ordered coordinates!
            ex)
                "location": [
					{
						"x8": 882,
						"y9": 236,
						"x9": 823,
						"x10": 741,
						"y1": 234,
						"x1": 741,
						"y2": 252,
						"y10": 234,
						"x2": 700,
						"y3": 283,
						"x3": 682,
						"y4": 316,
						"x4": 698,
						"y5": 344,
						"x5": 786,
						"y6": 351,
						"x6": 865,
						"y7": 316,
						"x7": 917,
						"y8": 264
					}
				],
            #result = [ int(val) for val in doc[0].values() ]
            '''

            # work around: ordered read keys (x and y pair) based on sequence number: ex) x1, y1, x2, y2, ...
            start = 1
            end = int(len(doc[0].values()) / 2) + start
            for index in range(start, end):
                result.append(int(doc[0][f"x{index}"]))
                result.append(int(doc[0][f"y{index}"]))
    elif tag == "boxlocation":  # for bounding box position
        if len(doc[0].values()) != 4:
            result = None
        else:
            result.append(int(doc[0]["x"]))
            result.append(int(doc[0]["y"]))
            result.append(int(doc[0]["width"]))
            result.append(int(doc[0]["height"]))
    elif tag == "resolution":
        result = [ int(val) for val in "".join(str(doc).split()).lower().split('x') ]
    else:
        result = doc
    
    return result


def onParseAnnotation(path, hint):
    '''
    onParseAnnotation
    ================

    hint for metadata structure:

        SET_MEATDATA_STRUCT = {
            'filename': ('metaData', 'Raw data ID'),
            'resolution': ('metaData', 'resolution'),
            'shape': ('labelingInfo', 'polygon', 'type'),
            'color': ('labelingInfo', 'polygon', 'color'),
            'location': ('labelingInfo', 'polygon', 'location'),
            'label': ('labelingInfo', 'polygon', 'label'),
            'box': ('labelingInfo', 'box', 'type'),
            'boxlocation': ('labelingInfo', 'box', 'location'),
            }
    '''

    annotations = {}

    try:
        with open(path, 'r', encoding='utf-8') as fjson:
            try:
                json_doc = json.load(fjson)
                
                for target in hint.keys():
                    annotations[target] = []

                    keys = hint[target]             # if target is 'location', keys contains ('labelingInfo', 'polygon', 'location')
                    temp_doc = json_doc[keys[0]]    # sub doc for first item in keys, ex) [ { "polygon": {}, "polygon": {}, ... } ]
                    
                    # for every hint, check if the value of the first key is a list
                    if isinstance(temp_doc, list):
                        count = len(temp_doc)
                        for index in range(count):
                            # retrieve one of items, ex) { "polygon": {}, ... }
                            # from json doc in forms of { "labelingInfo": [ { "polygon": {}, ... }, { "polygon": {}, ... }, ... ] }
                            temp_doc = json_doc[keys[0]][index]
                            temp_val = __getTerminalItems(target, temp_doc, keys[1:]) # ex) target is 'location'
                            if temp_val is not None:
                                annotations[target].append(temp_val)
                    else:
                        temp_val = __getTerminalItems(target, temp_doc, keys[1:])
                        if temp_val is not None:
                            annotations[target].append(temp_val)

                    # work around: for irregular file name in 'Raw data ID'
                    fname, _ = os.path.splitext(os.path.basename(path))
                    _, ext = os.path.splitext(annotations['filename'][0])
                    annotations['filename'] = [ fname + ext ]
                logout("annotations =\n{0}".format(annotations))

            except (ValueError, KeyError, JSONDecodeError, IndexError) as e:
                annotations = None
                logout("JSON formatted metadata parsing failed! {0}".format(e))
    except (FileNotFoundError, PermissionError) as e:
        annotations = None
        logout(e)

    return annotations


def checkSizeOrientationAlignmentForImgAndMeta(imgSize, sizeInMeta):
    logout(f"image size = {imgSize}, resolution in metadata = {sizeInMeta}")
    if len(imgSize) != 2 or len(sizeInMeta) != 2:
        return False

    # if size (w x h) of image equals to size in metadata, size and orientation matched.
    # otherwise, for example, image size is (w x h) and metadata is (h x w), size and orientation not matched.
    return tuple(imgSize) == tuple(sizeInMeta)


def checkBoundary(resolution, points):
    # check boundary
    for index in range(0, len(points) - 1, 2):
        if (points[index] < 0 or points[index] > resolution[0]) or (points[index + 1] < 0 or points[index + 1] > resolution[1]):
            logout("checkBoundary failed (exceed boundary)")
            return False
    
    return True


def checkAreaRatio(imgSize, ROIlist, minRatio):
    '''
    checkAreaRatio
    ===========================

    Checks whether the labeling area satisfies the minimum size.   

    Parameters:
    ---------------------
    - imgSize: Width and height information expressed in tuple or list format as image size (e.g. (w, h) or [w, h])
    - ROIlist: Position and size information expressed as a list or lists in dict for the labeled region (e.g. (left, top, right, bottom), (x1, y1, x2, y2, ...) or { key: (list of coordinates), ... })
    - minRatio: minimum acceptable ratio of the area of the labeling area to the total area of the image (value between 0.0 and 1.0)

    Returns:
    ---------------------
    Boolean
    - if the area of the labeling region is smaller than minRatio, return False.
    - otherwise, True.
    '''

    if len(imgSize) != 2 or (imgSize[0] <= 0 or imgSize[1] <= 0):
        logout(f"checkAreaRatio failed (The width and height of imgSize ({imgSize}) cannot be less than or equal to 0)")
        return False

    if minRatio < CONST_MIN_AREARATIO or minRatio > CONST_MAX_AREARATIO:
        logout(f"checkAreaRatio failed (minRatio out of range ({minRatio}) - must be between {CONST_MIN_AREARATIO} and {CONST_MAX_AREARATIO})")
        return False

    if minRatio == CONST_MIN_AREARATIO:
        return True

    imgArea = imgSize[0] * imgSize[1]

    def countInnerPixels(coords):
        innerPixels = -1
        counts = len(coords)
        if counts == 4:
            # bounding box style
            # (left, top, right, bottom)
            width = coords[2] - coords[0]
            height = coords[3] - coords[1]
            innerPixels = width * height
        elif counts >= 6 and counts % 2 == 0:
            # polygon style
            # (x1, y1, x2, y2, ...)
            x_list = coords[0::2]
            y_list = coords[1::2]
            innerPixels = 0.5*np.abs(np.dot(x_list,np.roll(y_list,1))-np.dot(y_list,np.roll(x_list,1)))

        return innerPixels

    # count inner-pixels
    if not isinstance(ROIlist, list) and not isinstance(ROIlist, tuple):
        logout("checkAreaRatio failed (labeled region info must be a list or a tuple)")
        return False

    innerPixels = countInnerPixels(ROIlist)
    if innerPixels == -1:
        logout("checkAreaRatio failed (invalid labeled region info)")
        return False

    innerPixelRatio = innerPixels / imgArea
    if innerPixelRatio < minRatio:
        logout(f"checkAreaRatio failed (small labeled region: inner pixels ({innerPixels}) / image ({imgArea}) = {innerPixelRatio} < {minRatio})")
        return False

    return True


def getBoundingBox(info, update = False):
    TAGS = ['shape', 'polygon', 'location', 6] if update == True else ['box', 'box', 'boxlocation', 4]
    bboxes = {}

    for index in range(len(info[TAGS[0]])):
        shape = info[TAGS[0]][index]
        if shape != TAGS[1]: continue

        location = info[TAGS[2]][index]
        if len(location) < TAGS[3] or len(location) % 2 != 0:
            logout("getBoundingBox failed (invlaid coordinates)")
            return False, CONST_ERROR_0007

        if update == True:
            # if update is True, calculate bbox info from polygon info
            bbox = [ info['resolution'][0][0], info['resolution'][0][1], 0, 0 ]
            for step in range(0, len(location) - 1, 2):
                x = location[step]
                y = location[step + 1]

                if checkBoundary( info['resolution'][0], (x, y) ) == False:
                    return False, CONST_ERROR_0002

                if bbox[0] >= x: bbox[0] = x
                if bbox[2] <= x: bbox[2] = x
                if bbox[1] >= y: bbox[1] = y
                if bbox[3] <= y: bbox[3] = y
                    
            bboxes[index] = bbox
        else:
            # if update is False, just retrieve existing bbox info and convert to position info
            # (x, y, width, height) --> position of region: (x, y, width + x, height + y)
            bboxes[index] = [ location[0], location[1], location[2] + location[0], location[3] + location[1] ]

    if len(bboxes) < 1:
        logout("getBoundingBox failed (no bounding boxes)")
        return False, CONST_ERROR_0008

    return bboxes, None


# TODO: foreground = 0, background = 1, contour = 2
'''
CONST_COLOR_BG = 127    # background
CONST_COLOR_FG = 255    # foreground
CONST_COLOR_CT = 0      # contour
CONST_LINE_WIDTH = 20   # contour-line width
'''
CONST_COLOR_BG = 0 #1    # background
CONST_COLOR_FG = 1 #0    # foreground
CONST_COLOR_CT = 1 #2      # contour
CONST_LINE_WIDTH = 2 #40   # contour-line width

def onConvPolygon2MaskImg(info, inputpath, output_target_path, config:Config):
    global CONST_FOLDER_MASK
    global CONST_FOLDER_PREVIEW

    global CONST_COLOR_BG
    global CONST_COLOR_FG
    global CONST_COLOR_CT

    results = []

    if info is None or not isinstance(info, dict) or len(info) == 0:
        logout("onConvPolygon2MaskImg failed (invalid info argument")
        return False, results, CONST_ERROR_0004

    if len(info['shape']) != len(info['location']):
        # TODO: collect invalid sample in 'ignored' folder
        logout("onConvPolygon2MaskImg failed (shape and location mismatched)")
        return False, results, CONST_ERROR_0005

    if 'polygon' not in info['shape']:
        logout("onConvPolygon2MaskImg failed (no polygon items in shape element)")
        return False, results, CONST_ERROR_0006

    path_mask = output_target_path + os.sep + CONST_FOLDER_MASK
    if os.path.exists(path_mask) == False:
        os.makedirs(path_mask)

    path_preview = output_target_path + os.sep + CONST_FOLDER_PREVIEW
    if os.path.exists(path_preview) == False:
        os.makedirs(path_preview)

    path_cropped_input = output_target_path + os.sep + CONST_FOLDER_INPUT
    if os.path.exists(path_cropped_input) == False:
        os.makedirs(path_cropped_input)

    fname, ext = os.path.splitext(info['filename'][0])
    fname = '{0}_'.format(fname) * (config.SET_DECO_FOR_PREFIX == False) + config.SET_FILENAME_MASKDECO + '_{0}'.format(fname) * config.SET_DECO_FOR_PREFIX

    path_cropped_input = path_cropped_input + os.sep + fname
    path_mask = path_mask + os.sep + fname
    path_preview = path_preview + os.sep + fname

    logout("p2m output result: {0}".format(path_mask + ext))

    filename = ""

    # TODO: consider generate separated images for each region
    try:
        with Image.open(inputpath + os.sep + info['filename'][0]) as img:
            if config.SET_ROTATE_WITH_EXIF == True:
                img = ImageOps.exif_transpose(img)
            preview_img = img.copy()
            mask_img = Image.new('L', img.size, CONST_COLOR_BG)

            for index in range(len(info['shape'])):
                shape = info['shape'][index]
                if shape != 'polygon': continue

                location = info['location'][index]
                size = len(location)
                if size < 6 or size % 2 != 0:
                    logout("onConvPolygon2MaskImg failed (invlaid coordinates)")
                    return False, results, CONST_ERROR_0007

                # for mask image
                ImageDraw.Draw(mask_img).polygon(location, outline=CONST_COLOR_CT, fill=CONST_COLOR_FG)
                ImageDraw.Draw(mask_img).line(location, fill=CONST_COLOR_CT, width=CONST_LINE_WIDTH, joint="curve")

                # for preview
                ImageDraw.Draw(preview_img).polygon(location, outline=CONST_COLOR_CT)
                ImageDraw.Draw(preview_img).line(location, fill=CONST_COLOR_CT, width=CONST_LINE_WIDTH, joint="curve")

            if not config.SET_CROP_MASK:
                # save cropped original image
                filename = path_cropped_input + ext
                img.save(filename, quality=CONST_IMG_QUALITY, subsampling=CONST_IMG_SUBSAMPLE)
                results.append(filename)

                # save copped mask image
                filename = path_mask + ext
                mask_img.save(filename, quality=CONST_IMG_QUALITY, subsampling=CONST_IMG_SUBSAMPLE)
                results.append(filename)

                # save cropped preview image
                if config.SET_CREATE_EXTRAS:
                    filename = path_preview + ext
                    preview_img.save(filename)
                    results.append(filename)
            else:
                for index in range(len(info['shape'])):
                    shape = info['shape'][index]
                    if shape != 'polygon': continue

                    location = info['location'][index]
                    x_list = location[0::2]
                    y_list = location[1::2]
                    
                    left = min(x_list)
                    right = max(x_list)

                    top = min(y_list)
                    bottom = max(y_list)

                    right_margin = config.SET_CROP_MARGINE
                    bottom_margin = config.SET_CROP_MARGINE

                    left_margin = random.randint(int(config.SET_CROP_MARGINE/10), config.SET_CROP_MARGINE)
                    if left_margin > left:
                        right_margin += (left_margin - left)
                        left_margin -= (left_margin - left)
                    top_margin = random.randint(int(config.SET_CROP_MARGINE/10), config.SET_CROP_MARGINE)
                    if top_margin > top:
                        bottom_margin += (top_margin - top)
                        top_margin -= (top_margin - top)

                    left -= left_margin
                    right += right_margin
                    if right > img.size[0]:
                        right = img.size[0]

                    top -= top_margin
                    bottom += bottom_margin
                    if bottom > img.size[1]:
                        bottom = img.size[1]

                    # save cropped original image
                    filename = "{0}_{1:03d}{2}".format(path_cropped_input, index, ext)
                    img.crop((left, top, right, bottom)).save(filename, quality=CONST_IMG_QUALITY, subsampling=CONST_IMG_SUBSAMPLE)
                    results.append(filename)

                    # save copped mask image
                    filename = "{0}_{1:03d}{2}".format(path_mask, index, ext)
                    mask_img.crop((left, top, right, bottom)).save(filename, quality=CONST_IMG_QUALITY, subsampling=CONST_IMG_SUBSAMPLE)
                    results.append(filename)

                    # save cropped preview image
                    if config.SET_CREATE_EXTRAS:
                        filename = "{0}_{1:03d}{2}".format(path_preview, index, ext)
                        preview_img.crop((left, top, right, bottom)).save(filename, quality=CONST_IMG_QUALITY, subsampling=CONST_IMG_SUBSAMPLE)
                        results.append(filename)
    except Exception as e:
        logout("onConvPolygon2MaskImg failed (unexpected error):\n{0}".format(e))
        if filename != "":
            results.append(filename)
        return False, results, CONST_ERROR_0011

    return True, results, None


def updateBBOXtoJSON(jsonPath, outputPath, bboxes, info, config:Config):
    results = []
    jsonUpdatePath = ""

    try:
        # Open JSON metafile with 'input' parameter
        with open(jsonPath, 'r', encoding='utf-8') as fjson:
            json_doc = json.load(fjson)

        # add or update bbox info in json_doc
        for index in bboxes:
            bbox = bboxes[index]

            # JSON metadata structure changed
            boundingBox = {"color": info['color'][index], "location" : {"x" : bbox[0], "y" : bbox[1], "width" : bbox[2], "height" : bbox[3]}, "label": info['label'][index], "type": "box"}
            if 'box' in json_doc['labelingInfo']:
                json_doc['labelingInfo']['box'] = boundingBox
            else:
                json_doc['labelingInfo'].append( {"box": boundingBox} )

        # add bounding box info to JSON file
        jsonfile, jsonext = os.path.splitext(os.path.basename(jsonPath))
        jsonfile = '{0}_'.format(jsonfile) * (config.SET_DECO_FOR_PREFIX == False) + config.SET_FILENAME_BBOXDECO + '_{0}'.format(jsonfile) * config.SET_DECO_FOR_PREFIX + jsonext
        jsonUpdatePath = outputPath + os.sep + jsonfile
        with open(jsonUpdatePath, 'w', encoding='utf-8') as fjson2:
            json.dump(json_doc, fjson2, ensure_ascii=False, indent="\t")
            logout("p2b updated output result: {0}".format(jsonUpdatePath))

        results.append(jsonUpdatePath)
    except Exception as e:
        logout("polygon to bbox failed! because:\n{0}".format(e))
        if jsonUpdatePath != "":
            results.append(jsonUpdatePath)
        return False, results, CONST_ERROR_0009
    
    return True, results, None


def cropBBOXtoImg(inputpath, outputPath, previewPath, bboxes, info, config:Config):
    results = []

    fname, ext = os.path.splitext(info['filename'][0])
    fname = '{0}_'.format(fname) * (config.SET_DECO_FOR_PREFIX == False) + config.SET_FILENAME_BBOXDECO + '_{0}'.format(fname) * config.SET_DECO_FOR_PREFIX

    target_path = ""

    try:
        with Image.open(inputpath + os.sep + info['filename'][0]) as img:
            # check rotation info
            # if rotated, but metadata resolution is not, then return False
            if config.SET_ROTATE_WITH_EXIF == True:
                img = ImageOps.exif_transpose(img)
            img_copy = img.copy()
            
            for index in bboxes:
                bbox = bboxes[index]
                filename = "{0}_{1:03d}{2}".format(fname, index, ext)
                target_path = outputPath + os.sep + filename
                img_copy.crop(bbox).save(target_path, quality=CONST_IMG_QUALITY, subsampling=CONST_IMG_SUBSAMPLE)
                results.append(target_path)
                logout("p2c cropped output result: {0}".format(target_path))

                if config.SET_CREATE_EXTRAS:
                    ImageDraw.Draw(img).rectangle(bbox, outline=(255, 0, 0), width=CONST_LINE_WIDTH)

            if config.SET_CREATE_EXTRAS:
                filename = fname + ext
                target_path = previewPath + os.sep + filename
                img.save(target_path)

                results.append(target_path)
                logout("p2b overlaid output result: {0}".format(target_path))
    except Exception as e:
        logout("polygon to bbox failed! because:\n{0}".format(e))
        if target_path != "":
            results.append(target_path)
        return False, results, CONST_ERROR_0010

    return True, results, None


def onConvPolygon2BBox(info, bboxes, inputpath, jsonPath, output_target_path, config:Config):
    '''
    onUpdateBBoxMetadata
    ====================   
    
    Parameters:
    ---------------
        - info: dict object with parsed metadata
        - inputpath: input path for image files
        - jsonPath: path for metadata file (JSON format)
        - outputpath: output path for updated metadata
        - deco: additional string for file name
        - prefix: prefix string for file name
        - update: if True, updates json metadata in jsonPath
        - crop: if True, crop and save bbox images

    Returns:
    ---------------
        Nothing

    Results:
    ---------------
        if update is True, updates JSON metafile with boundingBox elements in output path (sub-folder name applied)
        if crop is True, the cropped images will be created based on the calculated bounding box info
    '''
    global CONST_FOLDER_BBOX
    global CONST_FOLDER_JSON
    global CONST_FOLDER_PREVIEW

    results = []

    if not config.SET_UPDATE_BBOXINFO and not config.SET_CROP_BBOX:
        # nothing to do, just skip
        return True, results, CONST_ERROR_0003

    if info is None or not isinstance(info, dict) or len(info) == 0:
        logout("onConvPolygon2BBox failed (invalid info argument")
        return False, results, CONST_ERROR_0004

    if config.SET_UPDATE_BBOXINFO:
        if len(info['shape']) != len(info['location']):
            # TODO: collect invalid sample in 'ignored' folder
            logout("onConvPolygon2BBox failed (shape and location mismatched)")
            return False, results, CONST_ERROR_0005
    else:
        if len(info['box']) != len(info['boxlocation']):
            # TODO: collect invalid sample in 'ignored' folder
            logout("onConvPolygon2BBox failed (shape and location mismatched)")
            return False, results, CONST_ERROR_0005

    if config.SET_UPDATE_BBOXINFO:
        if 'polygon' not in info['shape']:
            logout("onConvPolygon2BBox failed (no bounding box items in shape element)")
            return False, results, CONST_ERROR_0006
    else:
        if 'box' not in info['box']:
            logout("onConvPolygon2BBox failed (no bounding box items in shape element)")
            return False, results, CONST_ERROR_0006

    path_bbox = output_target_path + os.sep + CONST_FOLDER_BBOX
    if os.path.exists(path_bbox) == False:
        os.makedirs(path_bbox)

    path_json = output_target_path + os.sep + CONST_FOLDER_JSON
    if config.SET_UPDATE_BBOXINFO:
        if os.path.exists(path_json) == False:
            os.makedirs(path_json)

    path_preview = output_target_path + os.sep + CONST_FOLDER_PREVIEW
    if os.path.exists(path_preview) == False:
        os.makedirs(path_preview)

    if config.SET_UPDATE_BBOXINFO:
        result, update_list, err_code = updateBBOXtoJSON(jsonPath, path_json, bboxes, info, config)
        results += update_list

        if result == False:
            return result, results, err_code
    
    if config.SET_CROP_BBOX:
        result, update_list, err_code = cropBBOXtoImg(inputpath, path_bbox, path_preview, bboxes, info, config)
        results += update_list

        if result == False:
            return result, results, err_code
    
    return True, results, None


def getOutputPath(inputPath, currentPath, outputPath, aggregate):
    subdir_name = ""
    output_target_path = ""

    if inputPath != currentPath:
        subdir_name = currentPath.replace(inputPath, "")
        
        # subdir_name starts with path delimiter ('\' or '/')
        output_target_path = outputPath + subdir_name
    if aggregate:
        subdir_name = os.path.basename(currentPath)
        output_target_path = outputPath + os.sep + subdir_name
    if not os.path.exists(output_target_path):
        os.makedirs(output_target_path)

    return output_target_path, subdir_name


def rollback(pathList):
    try:
        for path in pathList:
            if os.path.isfile(path):
                os.remove(path)
    except Exception as e:
        logout(f"failed to remove invalid output file: {path}")


def onConvertData(currentPath, jsonPath, output_target_path, config:Config):

    # --------------------- Step 1. parse metadata ---------------------

    annotation = onParseAnnotation(jsonPath, config.SET_MEATDATA_STRUCT)
    if annotation is None:
        return False, ERROR_JSON, CONST_ERROR_0001

    with Image.open(currentPath + os.sep + annotation['filename'][0]) as img:
        imgSize = (img.size[1], img.size[0]) if config.SET_ROTATE_WITH_EXIF else img.size

        if checkSizeOrientationAlignmentForImgAndMeta(imgSize, annotation['resolution'][0]) == False:
            return False, ERROR_JSON, CONST_ERROR_0012

    logout("start conversion for {0} (label count: polygon({1}), bbox({2})".format(annotation['filename'][0], len(annotation['location']), len(annotation['boxlocation'])))
    
    # --------------------- Step 2. generate mask image ---------------------

    if config.SET_CREATE_MASKIMG:
        logout("generate mask image...")

        # all or nothing - if inacceptable bbox in bboxes, discard current data
        for polygon in annotation['location']:
            if checkBoundary( annotation['resolution'][0], polygon ) == False:
                return False, ERROR_JSON, CONST_ERROR_0002
            if checkAreaRatio(imgSize, polygon, config.SET_MIN_AREA_RATIO) == False:
                return False, ERROR_JSON, CONST_ERROR_0013

        result, resultList, errid = onConvPolygon2MaskImg(annotation, currentPath, output_target_path, config)
        if result == False:
            rollback(resultList)
            return False, ERROR_MASK, errid

    # --------------------- Step 3. update and crop bounding box ---------------------

    if config.SET_UPDATE_BBOXINFO or config.SET_CROP_BBOX:
        logout("update bbox and/or crop image...")

        bboxes, err_code = getBoundingBox(annotation, config.SET_UPDATE_BBOXINFO)
        if bboxes == False:
            return False, ERROR_JSON, err_code

        # all or nothing - if inacceptable bbox in bboxes, discard current data
        for key in bboxes:
            if checkAreaRatio(imgSize, bboxes[key], config.SET_MIN_AREA_RATIO) == False:
                return False, ERROR_JSON, CONST_ERROR_0013

        result, resultList, errid = onConvPolygon2BBox(annotation, bboxes, currentPath, jsonPath, output_target_path, config)
        if result == False:
            rollback(resultList)
            return False, ERROR_BBOX, errid

    # --------------------- Step 4. finishing ---------------------
    
    # copy original image and json
    #not config.SET_CROP_MASK and 
    if config.SET_CREATE_EXTRAS:
        dest = output_target_path + os.sep + CONST_FOLDER_INPUT
        if os.path.exists(dest) == False:
            os.makedirs(dest)
        shutil.copy2(currentPath + os.sep + annotation['filename'][0], dest)
        shutil.copy2(jsonPath, dest)

    return True, None, None


def __searchDirFiles(prevProcessedList, config:Config):

    # start traverse directories
    for (path, dirs, files) in os.walk(config.SET_DATA_PATH):
        logout("search path: {0}\n\t{1} dirs: {2}\n\t{3} files found".format(path, len(dirs), dirs, len(files)))

        if len(files) > 0:
            output_target_path, subdir_name = getOutputPath(config.SET_DATA_PATH, path, config.SET_OUTPUT_PATH, config.SET_AGGREGATE_CLASSES)
            logout("target output path with sub-folder:\n\tsub-folder name = {0}\n\toutput target path = {1}".format(subdir_name, output_target_path))

        for jsonfile in files:
            # skip if the file extension is not ".json"
            _, ext = os.path.splitext(jsonfile)
            if ext != CONST_METADATA_EXT: continue

            # if set option --resume, skip if already processed
            jsonPath = path + os.sep + jsonfile
            if config.SET_RESUME and jsonPath in prevProcessedList:
                prevProcessedList.remove(jsonPath)
                logout(f"{jsonPath} already processed, skip...")
                continue

            result, tag, errid = onConvertData(path, jsonPath, output_target_path, config)

            # store fail info
            if result == False:
                __updateIgnoredList(jsonPath, tag, errid, config)

            # whether onConvertData failed or not, updates processed list
            # if process resumed with --resume option, failed data will be skipped.
            # in other words, if process failed files again, we need to use ignored list.
            __updateProcessedList(jsonPath, config) 

        if not config.SET_FIND_RECURSIVELY:
            logout("stop searching (--recursive option not set)")
            break
    
    __updateProcessedList("", config, True)
    __updateIgnoredList("", "", "", config, True)


def __serachForRework(prevProcessedList, config:Config):

    if not config.SET_REWORK or len(prevProcessedList) == 0:
        logout("re-work process failed (check option --rework and/or --reworkmode)")
        return

    # start traverse file list
    for entry in prevProcessedList:
        path, jsonfile = os.path.split(entry)
        logout(f"try to rework on {entry}")

        if os.path.exists(entry) and os.path.isfile(entry):
            output_target_path, subdir_name = getOutputPath(config.SET_DATA_PATH, path, config.SET_OUTPUT_PATH, config.SET_AGGREGATE_CLASSES)
            logout("target output path with sub-folder:\n\tsub-folder name = {0}\n\toutput target path = {1}".format(subdir_name, output_target_path))

        _, ext = os.path.splitext(jsonfile)
        if ext != CONST_METADATA_EXT: continue

        jsonPath = path + os.sep + jsonfile

        result, tag, errid = onConvertData(path, jsonPath, output_target_path, config)

        # store fail info
        if result == False:
            __updateIgnoredList(jsonPath, tag, errid, config)

        # whether onConvertData failed or not, updates processed list
        # if process resumed with --resume option, failed data will be skipped.
        # in other words, if process failed files again, we need to use ignored list.
        __updateProcessedList(jsonPath, config)                
    
    __updateProcessedList("", config, True)
    __updateIgnoredList("", "", "", config, True)


def onSearchDir(config:Config):
    logout("\n\nStart search and conversion...\n\n")

    # ready to resume
    prevProcessedList = []
    if config.SET_RESUME or config.SET_REWORK:
        prevProcessedList = __getPrevProcessedList(config)
    if config.SET_REWORK:
        prevIgnoredList = __getPrevIgnoredList(config)
        if config.SET_REWORK_MODE == 'i':
            prevProcessedList = prevIgnoredList
        elif config.SET_REWORK_MODE == 'p-i' or config.SET_REWORK_MODE == 'a':
            for entry in prevIgnoredList:
                target = entry.split('|')[0]
                if target in prevProcessedList:
                    prevProcessedList.remove(target)

    try:
        if not config.SET_REWORK:
            __searchDirFiles(prevProcessedList, config)
        else:
            __serachForRework(prevProcessedList, config)
    except Exception as e:
        logout("search failed:\n{0}\n{1}".format(e.__class__, e))
        pass


def doConvert(config:Config):
    global CONST_METADATA_EXT

    logout("process being started.", True)

    # check input path
    if not os.path.exists(config.SET_DATA_PATH):
        logout("ERROR! input path not exist! check --input option:\n\t{0}".format(config.SET_DATA_PATH), True, True)
        return
    
    if not os.path.exists(config.SET_OUTPUT_PATH):
        os.makedirs(config.SET_OUTPUT_PATH)
        logout("output path created: {0}".format(config.SET_OUTPUT_PATH))

    if os.path.isdir(config.SET_DATA_PATH):
        # for directory
        onSearchDir(config)
    else:
        # for single file
        result, tag, errid = onConvertData(os.path.dirname(config.SET_DATA_PATH), config.SET_DATA_PATH, config.SET_OUTPUT_PATH, config)
        if result == False:
            __updateIgnoredList(config.SET_DATA_PATH, tag, errid, config, True)
        __updateProcessedList(config.SET_DATA_PATH, config, True)

    logout("process done.", True, True)


if __name__ == "__main__":
    args = __getArguments()
    config = __parseArguments(args)

    doConvert(config)
