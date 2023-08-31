# A Study of Ball and Player Detection System in Paddle Tennis with Deep Neural Networks

This is the source repository for the Master Thesis of the Master in Modelling for Science and Engineering, supervised by Dr. Sundus Zafar. Image and video data is not uploaded here for storage limitations reasons.

## Directory structure

The relevant files to understand the implemntation of the model are:

- `model_ball_detection`:
  - `dataset.py`: contains the dataset class.
  - `model.py`:  contains the Resnet50 model definition.
  - `train.py`: contains the training code.
  - `datapoint_class.py`: contains the datapoint class. This is a wrapper type over the `numpy.array` type, that contains method utilities.
  - `prediction.py`: contains the code for prediction. In particular.
- `ball_detection_contour`: 
  - `main.ipynb`: the file contains the main function for the pipeline and the code needed to create the ball positions, directions and player heatmap figures.