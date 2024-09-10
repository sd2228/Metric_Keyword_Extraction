import os

ROOT_DIR = os.path.dirname(os.getcwd()) # Getting the root directory of the project (parent directory of the current working directory)

SRC_DIR = os.path.join(ROOT_DIR + '/Keyword')


# Creating paths for various data directories
DATA_DIR = os.path.join(SRC_DIR + '/data')
OUTPUT_DIR_WEEK = os.path.join(SRC_DIR + '/output_week')
OUTPUT_DIR_SINGLE= os.path.join(SRC_DIR + '/output_single')

list_dir = [DATA_DIR,OUTPUT_DIR_WEEK,OUTPUT_DIR_SINGLE]

for x in list_dir:
    if not os.path.exists(x):
        os.makedirs(x)