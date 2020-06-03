#-*- coding: utf8 -*-
import os, logging

# paths
DATA_DIR = '/mnt/fastdata/kaggle/siim-isic-melanoma-classification/'
WORK_DIR = '/mnt/fastdata/kaggle/workdir'
RESULT_DIR = os.path.join(WORK_DIR, 'results')
if not os.path.exists(WORK_DIR): os.mkdir(WORK_DIR)
if not os.path.exists(RESULT_DIR): os.mkdir(RESULT_DIR)


TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'jpeg')

# logger
logger = logging.getLogger('app')
fh = logging.FileHandler(os.path.join(RESULT_DIR, 'application.log'))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
