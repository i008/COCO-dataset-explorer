# Running the explorer on example data 

You can use the predictions i prepared and explore the results on the coco validation dataset
the predictions are coming from a Mask RCNN model trained with mmdetection.

### 1. download (extract) the labels, annotations and images:

images:  
http://images.cocodataset.org/zips/val2017.zip  
predictions file:     
https://i008data.s3.eu-central-1.amazonaws.com/coco_val_results_maskrcnn50.json.segm.json  
labels:    
https://i008data.s3.eu-central-1.amazonaws.com/instances_val2017.json  

### 2. run 
```bash
streamlit run  coco_explorer.py -- --coco_train /Users/jcieslik/Downloads/annotations/instances_val2017.json --coco_predictions /Users/jcieslik/coco_val_results_maskrcnn50.json.segm.json  --images_path /Users/jcieslik/Downloads/val2017/
```



# Running on your own data

In the same way you can explore your own results. Just follow the official COCO dataset format for annotations and predictions.