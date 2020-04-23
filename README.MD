# Running the demo on example data

### download (extract) the labels, annotations and images:

images:  
http://images.cocodataset.org/zips/val2017.zip  
predictions file:     
https://i008data.s3.eu-central-1.amazonaws.com/coco_val_results_maskrcnn50.json.segm.json  
labels:    
https://i008data.s3.eu-central-1.amazonaws.com/instances_val2017.json  

### run 
```bash
streamlit run  coco_explorer.py -- --coco_train /Users/jcieslik/Downloads/annotations/instances_val2017.json --coco_predictions /Users/jcieslik/coco_val_results_maskrcnn50.json.segm.json  --images_path /Users/jcieslik/Downloads/val2017/
```


