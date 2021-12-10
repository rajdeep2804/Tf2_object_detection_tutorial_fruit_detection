# Tf2_object_detection_tutorial_fruit_detection

## Description
 Objective of this repo is to explain training process for tf2 and various state-of-the-art object-detection models (Faster R-CNN, R-FCN and SSD) combined with various feature extractors (Resnet V1 50, Resnet V1 101, Inception V2, Inception Resnet V2 and Mobilenet V1) with custom data for object detection. various publicly available object-detection models that were pre-trained on the Microsoft COCO dataset can be fine-tuned on custom dataset.

## Testing CUDA,
To test that CUDA works, go to the CUDA demo suite directory:
`cd /usr/local/cuda/extras/demo_suite/`
./deviceQuery

## Tensorflow object detection api

## Requirements

-Python 3.8.1, Tensorflow-gpu 2.4.1 and cuda 11.5
-Tensorflow repositories: [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [Tensorflow object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

## Pretrained models
You can download model weights of your choice from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Unzip it to the models dir.

- [faster_rcnn_inception_v2](https://drive.google.com/open?id=1LRCSWIkX_i6ijScMfaxSte_5a_x9tjWF)
- [faster_rcnn_resnet_101](https://drive.google.com/open?id=15OxyPlqyOOlUdsbUmdrexKLpHy1l5tP9)
- [faster_rcnn_resnet50](https://drive.google.com/open?id=1aEqlozB_CzhyJX_PO6SSiM-Yiv3fuO8V)
- [rfcn_resnet101](https://drive.google.com/open?id=1eWCDZ5BxcEa7n_jZmWUr2kwHPBi5-SMG)
- [ssd_inception_v2](https://drive.google.com/open?id=1TKMd-wIZJ1aUcOhWburm2b6WgYnP0ZK6)
- [ssd_mobilenet_v1](https://drive.google.com/open?id=1U31RhUvE1Urr5Q92AJynMvl-oFBVRxxg)

## Preparing pipeline for initializing training.
* Convert XML to CSV format using 'xml_to_csv.py' in data_gen dir. 
train : 
```bash 
python3 xml_to_csv.py --annot_dir data_images/train --out_csv_path train_labels.csv
```
test : 
```bash 
python3 xml_to_csv.py --annot_dir data_images/test --out_csv_path test_labels.csv
```
* Genrating TFrecords using 'generate_tfrecord.py' in data_gen dir.
train : 
```bash 
python3 generate_tfrecord.py --path_to_images data_images/train --path_to_annot train_labels.csv --path_to_label_map fruit.pbtxt --path_to_save_tfrecords train.records
```
test : 
```bash 
python3 generate_tfrecord.py --path_to_images data_gen/data_images/test --path_to_annot data_gen/test_labels.csv --train  path_to_label_map data_gen/fruit.pbtxt --path_to_save_tfrecords data_gen/test.records
```
* Download the corresponding original config file from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2). eg [ssd_mobilenet_v2_320x320_coco17_tpu-8.config](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config) and make the following changes based on your use case:
* Used `num_classes: 3` based on number of classes, instead of 90 classes in coco dataset.
* Changed `fine_tune_checkpoint_type: "classification"` to `fine_tune_checkpoint_type: "detection"` as we are using the pre-trained detection model as initialization.
* Added the path of the pretrained model in the field `fine_tune_checkpoint:`, for example using the mobilenet v2 model `fine_tune_checkpoint: "../models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"`  
* Changed `batch_size: 512` to a reasonable number based on GPU memory like `batch_size: 16`
* Added the maximum number of training iterations in `num_steps:`, and also used the same number in `total_steps:`
* Adapted the learning rate to our model and batch size (originally they used higher learning rates because they had bigger batch sizes). This values needs some testing and tuning:
    ``` 
    cosine_decay_learning_rate {
        learning_rate_base: 0.025
        total_steps: 15000
        warmup_learning_rate: 0.005
        warmup_steps: 300 }
    ```
* The `label_map_path:` should point labelmap file `label_map_path: "../models/raccoon_labelmap.pbtxt"`
* You need to set the `tf_record_input_reader` under both `train_input_reader` and `eval_input_reader`. This should point to the tfrecords we generated (one for training and one for validation).
    ```
    train_input_reader: {
        label_map_path: "../data/fruit.pbtxt"
        tf_record_input_reader {
            input_path: "../data/train.record"
        }
    }
    ``` 
* Prepare the labelmap according to your data, the [labelmap file](models/raccoon_labelmap.pbtxt) contains:

```
item {
  id: 1
  name: 'class name'
}
```
## Initialize training
* Once configuration file if prepared, start the training by typing the following commands:
```bash 
python3 model_main_tf2.py --pipeline_config_path ssd_mobilenet_v2.config --model_dir training/train/ --alsologtostderr
```
* Evaluating Model performance : 
```bash
python3 model_main_tf2.py --pipeline_config_path ssd_mobilenet_v2.config --model_dir training/train --checkpoint_dir training/train
```
Note that running the evaluation script along with the training requires another GPU dedicated for the evaluation. So, if you don't have enough resources, you can ignore running the validation script, and run it only once when the training is done. However, you can run the evaluation on the CPU, while the training is running on the GPU. Simply by adding this flag before running the evaluation script export `CUDA_VISIBLE_DEVICES="-1"`, which makes all the GPUs not visible for tensoflow, so it will use the CPU instead.
* Visualising model on Tensorboard :
`cd training/train/eval`
```bash
tensorboard --port 6004 --logdir=
```
## Exporting your trained model for inference
When the training is done, Tensorflow saves the trained model as a checkpoint. Now we will see how to export the models to a format that can be used for inference, this final format usually called saved model or frozen model.
```bash 
python3 exporter_main_v2.py --input_type="image_tensor" --pipeline_config_path=ssd_mobilenet_v2.config --trained_checkpoint_dir=training/train --output_directory=training/weights
```
The weights directory will contain 2 folders; saved_model and checkpoint. The saved_model directory contains the frozen model, and that is what we will use for inference. The checkpoint contains the last checkpoint in training, which is useful if you will use this model as pretrained model for another training in the future, similar to the checkpoint we used from the pretrained model with coco dataset.

## Setup for inferencing different models
- Clone this git repo, go to `Tensorflow -> Models -> Reaseach -> Object_detection dir`, and Download the object_detection.
- Make folder by name `Model` and download the weight files given above in this dir.
- Make folder by name `images`, where you can place all the test images for inference.
- Run `Object_detection_tf2` notebook for inferencing the images.