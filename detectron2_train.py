#Script that trains detectron2 models using .yaml file, need path for coco json and image folder

#import packages
import detectron2
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from model_3_save_info import Model_Info 
import argparse

class Train_Detectron2_Model():

    def register_dataset(self, coco_json_path, image_path):
        #Register dataset
        from detectron2.data.datasets import register_coco_instances
        register_coco_instances("UCMerced_dataset_train", {}, coco_json_path, image_path) 

        my_dataset_train_metadata = MetadataCatalog.get("UCMerced_dataset_train")
        dataset_dicts = DatasetCatalog.get("UCMerced_dataset_train")

        return my_dataset_train_metadata, dataset_dicts

    def train(self, cfg, info):
        #Train!
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("UCMerced_dataset_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.0005
        cfg.SOLVER.MAX_ITER = 20000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        # cfg.MODE_MASK = True

        info.save_model_settings(cfg) #saves the yaml file for all parameters of the model

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()

    def main(self, coco_json_path, image_path):
        cfg = get_cfg()

        info = Model_Info()
        
        info.save_category_names(coco_json_path) #saves all category names in separate json

        my_dataset_train_metadata, dataset_dicts = self.register_dataset(coco_json_path, image_path)

        self.train(cfg, info)

if __name__ == "__main__":

    train = Train_Detectron2_Model()

    parser = argparse.ArgumentParser(description="Train model with Detectron2")
    parser.add_argument("--coco_json", type=str, dest="coco_json_path", required=True, help="The path to the coco json that will be \
                        used.")
    parser.add_argument("--image_folder", type=str, dest="image_path", required=True, help="The path to the image folder that will be \
                        used.")

    args = parser.parse_args()

    train.main(args.coco_json_path, args.image_path) #need to define path for both json and folder where images are stored
