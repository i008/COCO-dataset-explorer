### What is this

This tool given a coco annotations file and coco predictions file will let you explore your dataset, visualize results
and calculate important metrics.


### Running the explorer on example data 

You can use the predictions i prepared and explore the results on the coco validation dataset
the predictions are coming from a Mask RCNN model trained with mmdetection.

1. download (and extract) the labels, annotations and images:

https://drive.google.com/open?id=1a0kNT_UXhU7FzljmIdQIBECRCUMy4PPs

2. setup using docker
```bash
sudo docker run -p 8501:8501 -it -v {path_to_downloaded_folder from point 1.}/coco_results:/coco_results i008/coco_explorer streamlit run  coco_explorer.py -- --coco_train /coco_results/instances_val2017.json --coco_predictions /coco_results/coco_val_results_maskrcnn50.json.segm.json  --images_path /coco_results/val2017/
```

2. setup using conda
```bash
conda env update
conda activate cocoexplorer
streamlit run coco_explorer.py -- --coco_train /Users/jcieslik/Downloads/annotations/instances_val2017.json --coco_predictions /Users/jcieslik/coco_val_results_maskrcnn50.json.segm.json  --images_path /Users/jcieslik/Downloads/val2017/
```
3.
go to localhost:8501


### Running on your own data

In the same way you can explore your own results. Just follow the official COCO dataset format for annotations and predictions.


### Examples

![alt text](./static/demo1.png "Logo Title Text 1")





![alt text](./static/demo2.png "Logo Title Text 1")
