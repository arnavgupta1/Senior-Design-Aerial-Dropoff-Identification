#literally a script to move all uc merced land images to one folder

import shutil
import os

all_img_path = "/Users/arnavgupta/Desktop/Documents/BU/Fall 2021/ME460 - Senior Design/UCMerced_LandUse/png_images/"
dest_dir = "/Users/arnavgupta/Desktop/Documents/BU/Fall 2021/ME460 - Senior Design/UCMerced_LandUse/images_ref"

for folder in os.listdir(all_img_path):
    if folder == ".DS_Store":
        continue
    folder_path = all_img_path + folder
    print ("folder : " + folder)
    for infile in os.listdir(folder_path):
        print ("file : " + infile)

        file_path = os.path.join(folder_path, infile)
        shutil.copy(file_path, dest_dir)

