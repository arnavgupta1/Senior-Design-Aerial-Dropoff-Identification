import json
import os
from PIL import Image
import cv2

#Format for Image json section:
#{"id": 2075, "width": 767, "height": 575, "file_name": "b404c7f8-f164-4f62-913d-3f614cdefa91.jpg"}

#Format for Annotation json section
#{"id": 13480, "category_id": 1, "iscrowd": 0, "segmentation": [[944.68, 609.04, 951.42, 609.42]], 
# "image_id": 520, "area": 11366.834899999201, "bbox": [806.0, 462.0, 145.0, 150.0]}

f = open('UCMerced_LandUse_annotations.json', 'r+')

data = json.load(f)
all_img_path = "/Users/arnavgupta/Desktop/Documents/BU/Fall 2021/ME460 - Senior Design/UCMerced_LandUse/png_images/"
images = {"images": []} #declare image section of the json file, array has list of dictionaries containing info about all imgs
annotations = {"annotations": []} #declare annotation section of the json file, annotation is complete image
category_ids = {"agricultural": 1, "airplane": 2, "baseballdiamond": 3, "beach": 4, "buildings": 5, "chaparral": 6,
                "denseresidential": 7, "forest": 8, "freeway": 9, "golfcourse": 10, "harbor": 11, "intersection": 12,
                "mediumresidential": 13, "mobilehomepark": 14, "overpass": 15, "parkinglot": 16, "river": 17, "runway": 18,
                "sparseresidential": 19, "storagetanks": 20, "tenniscourt": 21}
count_id = 1
for folder in os.listdir(all_img_path):
    if folder == ".DS_Store":
        continue
    folder_path = all_img_path + folder
    print ("folder : " + folder)
    for infile in os.listdir(folder_path):
        print ("file : " + infile)

        filepath = cv2.imread(folder_path + "/" + infile)
        img_path = folder_path + "/" + infile
        img = Image.open(img_path)
        width = img.width
        height = img.height 

        curr_img_dict = {"id": count_id, "width": width, "height": height, "file_name": infile}
        # images["images"][count_id] = curr_img_dict
        images["images"].append(curr_img_dict)

        curr_annotation_dict = {"id": count_id, "category_id": category_ids[folder], "iscrowd": 0, 
                                "segmentation": [[0, width, 0, height]], "image_id": count_id, "area": width*height, 
                                "bbox": [0, width, 0, height]}
        # annotations["annotations"][count_id] = curr_annotation_dict
        annotations["annotations"].append(curr_annotation_dict)
        count_id += 1



        
# data["images"] = images
# data["annotations"] = annotations
f.seek(0)
data.update(images)
data.update(annotations)
json.dump(data, f, indent=4)
print(data)


f.close()