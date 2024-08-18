# Recurrent Deep Learning-based Monthly Deforestation Prediction v2

### \<Scripts\>
- __main.py__ contains all processing steps
  - __main_GoogleColab.ipynb__ is a version of main.py for Google Colab

### \<Note\>
This repository is a modified version of [**the repository**](https://github.com/aistairc/Recurrent_deep_learning_based_monthly_deforestation_prediction) with the following changes: <br/>
 1. Improved flexibility of whole scripts
 2. Added config function for managing variables
 3. Added new feature from road information
 4. Added regression scenario (but not effective)
 5. Separated training and evaluation steps
 6. Added future predicition function
 7. Added two masking functions (based on output probability and sum of deforestated area in a mesh)
 8. Added EarlyStopping function
 9. Added save function for geojson file to visualize results in QGIS software
 10. Added an ipynb file that works with Google Colaboratory <br/>

### \<Overview\>
- This repository
  - Creates $1\times1$ km meshes of eight areas (Porto Velho, Humaita, Altamira, Vista Alegre do Abuna, Novo Progresso, Sao Felix do Xingu, S6W57, and S7W57) from preprocessed about five-year data (October 2017 to September 2022) using the Real-Time Deforestation Detection System (DETER)
  - Constructs recurrent deep learning-based prediction models consisting of two long short term memory (LSTM) layers and two fully-connected (FC) layers. In the case of the classification model, it has an additional softmax layer.
- You can train the mesh-wise model and evaluate its performance by applying an incremental training and testing method for two-year data (October 2020 to September 2022).<br/>
  - __Input features (in data/AreaName/feature)__:
      1. Binary of whether deforestation occurred or not (deforestation_event.csv)
      2. Area where deforestation occurred (deforestation_square_meter.csv)
      3. Ditance to the closest road (distance_to_closest_road_meter.csv)<br />
  - __Input sequence length__: 12 months<br />
- The deforestation prediction performances of the recurrent deep learning-based models for $1\times1$ km meshes with 1-month resolution averaged over eight areas in Brazil Amazon are summarized in <a href="https://github.com/aistairc/Recurrent_deep_learning_based_monthly_deforestation_prediction_v2/blob/main/model_performance.png" target="_blank">model_performance.png</a>.
- A comparative analysis of actual and predicted deforestation events from June to September in both 2021 and 2022 at Altamira is shown in <a href="https://github.com/aistairc/Recurrent_deep_learning_based_monthly_deforestation_prediction_v2/blob/main/comparative_analysis.jpg" target="_blank">comparative_analysis.jpg</a>.
- The results of the prediction of future information for each area using the latest trained model are shown in <a href="https://github.com/aistairc/Recurrent_deep_learning_based_monthly_deforestation_prediction_v2/blob/main/future_prediction.jpg">future_prediction.jpg</a>. In this case, the model is trained with features up to August 2022 and labels up to September 2022, and features from September 2022 to the past year are input to predict the labels for October 2022.

### \<Environments\>
- checked OS: WSL2 (Ubuntu 22.04)
- Anaconda 3
- Python 3.8.18
- PyTorch 2.2.0 + cuda 11.8
- scikit-learn 1.3.2
- skorch 0.15.0
- The virtual environment configuration is provided as requirements.txt
  - You can reproduce the environment by<br/>
```conda install --yes --file requirements.txt```<br/>
