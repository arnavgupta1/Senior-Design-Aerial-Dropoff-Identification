import cv2, os
from osgeo import gdal

base_path = "/Users/arnavgupta/Desktop/Documents/BU/Fall 2021/ME460 - Senior Design/UCMerced_LandUse/Images/"
new_path = "/Users/arnavgupta/Desktop/Documents/BU/Fall 2021/ME460 - Senior Design/UCMerced_LandUse/png_images/"
   
options_list = [
    '-ot Byte',
    '-of PNG',
    '-b 1',
    '-b 2',
    '-b 3',
    '-colorinterp_1 red',
    '-colorinterp_2 green',
    '-colorinterp_3 blue',
    '-scale'
]   

options_string = " ".join(options_list)

for folder in os.listdir(base_path): #iterate through folders of all categories
    if folder == ".DS_Store":
        continue
    folder_path = base_path + folder + "/"
    print ("folder : " + folder)
    new_png_path = os.path.join(new_path, folder)
    os.mkdir(new_png_path) #create folder of category in the png images folder
    for infile in os.listdir(folder_path): #iterate through all images within each category folder
        print("File: " + infile)
        outfile = infile.split('.')[0] + '.png'
        save_image_path = os.path.join(new_png_path, outfile)
        geo_image_path = folder_path + infile
        gdal.Translate( #convert image to png file format
            save_image_path,
            geo_image_path,
            options=options_string
        )
