# Senior-Design-Aerial-Dropoff-Identification

This is the code repository aiming to implement an object detection model to identify drop off locations for a pharmaceutical delivery service as a part of a Senior Design group at Boston University. This will aid our project in delivering our product to the proper location and be an integral part of our vision system. Our motivation for creating a drone to deliver pharmaceuticals was because of a lack of delivery routes to residences in rural areas as well as the lack of access to medical supplies since the number of pharmacies has been steadily decreasing in rural areas.

# Repository Organization

- Scripts contain all scripts used to modify data and create metadata
- [Detectron2](https://github.com/facebookresearch/detectron2) script contains training code for open source Detectron2 model meant for UC Merced Land Use dataset
- MLP_UCMercedLandUse is a notebook containing preprocessing, training, and testing of the final model used for the project
- Results Folder
    - Final_Model.pth is the final trained model that achieved highest accuracy while running inference
    - Sheet tracking iterations of model hyperparameters and data transformations with accuracy achieved with each iteration

# How to Run

- Open MLP_UCMercedLandUse.ipynb and download Results/Final_Model.pth
- [Download PNG version of UC Merced Land Use dataset](https://drive.google.com/drive/folders/15U4qIUKTZmD7lPufNeRKV6nQCDW6L7RP?usp=sharing)

Access trained model by either:
1. Training from scratch running imports, data import & transform, first model definition & initialization, training code sections
2. Loading saved model within Save and Load model section

Run Inference on model by opening inference section and:
1. Test accuracy of dataset on random batch of images in test set
2. Test accuracy of model on overall test set
3. Test accuracy on all sparseresidential images in test set
