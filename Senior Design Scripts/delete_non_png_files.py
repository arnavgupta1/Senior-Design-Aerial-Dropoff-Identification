import os

base_path = "/Users/arnavgupta/Desktop/Documents/BU/Fall 2021/ME460 - Senior Design/UCMerced_LandUse/png_images/"

for folder in os.listdir(base_path): #iterate through folders of all categories
    if folder == ".DS_Store":
        continue
    folder_path = base_path + folder + "/"
    print ("folder : " + folder)
    for infile in os.listdir(folder_path): #iterate through all images within each category folder
        outfile = infile.split('.')
        print(outfile)
        if len(outfile) > 2:
            print("is there any ?")
            os.remove(os.path.join(folder_path, infile))