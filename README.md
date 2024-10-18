# Semantic-Segmentaton
DL-based methods for semantic segmentation of 2D medical imaging dataset

**File Structure**
  - _Autoencoder_Model+Train.py_ trains a CNN-based autoencoder architecture for semantic segmentation
  - _model.py_ consists of a dilated UNet architecture used for semantic segmentation
  - _datasets.py_ defines the custom dataset and the transformations that will be performed on train and validation sets
  - _losses.py_ consists of class of a loss function that combines BCE and DICE loss along with an optional connected component loss
  - _Train.py_ trains the UNet model using the combined loss function
