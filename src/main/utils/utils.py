import os

def make_dir_if_not_exist(path):
    os.makedirs(path, exist_ok=True)