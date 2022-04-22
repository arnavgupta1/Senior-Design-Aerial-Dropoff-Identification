# Senior-Design-Aerial-Dropoff-Identification

This is the code repository aiming to implement an object detection model to identify drop off locations for a pharmaceutical delivery service as a part of a Senior Design group at Boston University. This will aid our project in delivering our product to the proper location and be an integral part of our vision system. Our motivation for creating a drone to deliver pharmaceuticals was because of a lack of delivery routes to residences in rural areas as well as the lack of access to medical supplies since the number of pharmacies has been steadily decreasing in rural areas.

# Code Organization

- Scripts contain all scripts used to modify data and create metadata
- [Detectron2](https://github.com/facebookresearch/detectron2) script contains training code for open source Detectron2 model meant for UC Merced Land Use dataset
- MLP_UCMercedLandUse is current model used for identifying dropoff locations from aerial images (work in progress)