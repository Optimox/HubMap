# HubMap
Kaggle HubMap


Basic pipeline so far.

Almost full pipeline implemented so far (except prediction inside Kaggle Notebook)


### How to get started

- clone the repo
- environment : no specific docker has been attached as I work with the official kaggle docker image
- Download data from Kaggle and put in HubMap/input/hubmap-kidney-segmentation
- Create PNG dataset using HubMap/notebooks/DatasetGenerator.ipynb
- train models : HubMap/notebooks/training_pipeline.ipynb
- whole slide prediction and scores : /HubMap/notebooks/prediction_pipeline.ipynb