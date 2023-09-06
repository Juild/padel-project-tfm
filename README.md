# A Study of Ball and Player Detection System in Paddle Tennis with Deep Neural Networks

This is the source repository for the Master Thesis of the Master in Modelling for Science and Engineering, supervised by [Dr. Sundus Zafar](https://www.linkedin.com/in/sunduszafar/). Image and video data used for training is not uploaded here for storage limitations reasons.

[Written report](https://github.com/Juild/padel-project-tfm/blob/main/TFM-4.pdf)

## Directory structure

The relevant files to understand the implemntation of the model are:

- `model_ball_detection`:
  - `dataset.py`: contains the dataset class.
  - `model.py`:  contains the Resnet50 model definition.
  - `train.py`: contains the training code.
  - `datapoint_class.py`: contains the datapoint class. This is a wrapper type over the `numpy.array` type, that contains method utilities.
  - `prediction.py`: contains the code for prediction. In particular get
- `ball_detection_contour`: `get_ball_prediction_confidence_per_image` is the one used in the main pipeline for predicting the region candidates for the ball.
  - `main.ipynb`: the file contains the main function for the pipeline and the code needed to create the ball positions, directions and player heatmap figures.
 
## Results

### Video

https://github.com/Juild/padel-project-tfm/assets/74079422/3573649b-5b21-48d0-9483-eef29791e07b

### Statistics

#### Player positions heatmap

<img width="789" alt="Screenshot 2023-09-06 at 10 50 34" src="https://github.com/Juild/padel-project-tfm/assets/74079422/192905ce-a641-469a-bae5-901f9a63339b">

#### Ball positions and most played directions


<img width="591" alt="Screenshot 2023-09-06 at 10 50 22" src="https://github.com/Juild/padel-project-tfm/assets/74079422/da495bf4-5a93-4f06-889e-ec5cca5db478">
