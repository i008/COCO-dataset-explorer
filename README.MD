### What is this

This tool given a COCO annotations file and COCO predictions file will let you explore your dataset, visualize results
and calculate important metrics.


### Running the explorer on example data 

You can use the predictions I prepared and explore the results on the COCO validation dataset.
The predictions are coming from a Mask R-CNN model trained with mmdetection.

1. Download (and extract in project directory) the labels, annotations and images:

https://drive.google.com/open?id=1wxIagenNdCt_qphEe8gZYK7H2_to9QXl

2. Setup using docker

```sh
sudo docker run -p 8501:8501 -it -v "$PWD"/coco_data:/coco_data i008/coco_explorer  \
    --coco_train /coco_data/ground_truth_annotations.json \
    --coco_predictions /coco_data/predictions.json  \
    --images_path /coco_data/images/
```

2. Setup using conda
```sh
conda env update
conda activate cocoexplorer
streamlit run coco_explorer.py -- --coco_train ./coco_data/ground_truth_annotations.json --coco_predictions ./coco_data/predictions.json  --images_path ./coco_data/val2017/
```
3 - go to localhost:8501


### Running on your own data

In the same way you can explore your own results. Just follow the official COCO dataset format for annotations and predictions.


### Examples

![alt text](./static/demo1.png "Logo Title Text 1")





![alt text](./static/demo2.png "Logo Title Text 1")
