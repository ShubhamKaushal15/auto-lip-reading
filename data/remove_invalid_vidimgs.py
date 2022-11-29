import os
import shutil

def getInvalid(dirPath = os.path.join('grid_imgs')):
    """
    removes video directories with less than num_required imgs/frames
    dirPath: path to the folder with videos s.t. grid_imgs/s{i}/abcd1e/0**.png
    """
    num_required = 60
    for root, dirs, files in os.walk(dirPath, topdown=False):
        for name in dirs:
            subDirPath = os.path.join(root, name)
            if len(os.listdir(subDirPath)) <= num_required:
                # shutil.rmtree(subDirPath)
                print(f"{subDirPath} removed")
            
                    
if __name__ == '__main__':
    getInvalid()