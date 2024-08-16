# Recurrent Deep Learning-based Monthly Deforestation Prediction v2

This repository is a modified version of [**the repository**](https://github.com/aistairc/Recurrent_deep_learning_based_monthly_deforestation_prediction) with the following changes: <br/>
 1. improved flexibility of whole scripts
 2. added config function for managing variables
 3. added new feature from road information
 4. added regression scenario (but not effective)
 5. separated training and evaluation steps
 6. added future predicition function
 7. added two masking functions (based on output probability and sum of deforestated area in a mesh)
 8. added EarlyStopping function
 9. added ipynb files that work with Google Colaboratory <br/>

### \<Overview\>
- Python scripts in this repository create $1\times1$ km meshes of eight areas (Porto Velho, Humaita, Altamira, Vista Alegre do Abuna, Novo Progresso, Sao Felix do Xingu, S6W57, and S7W57) from preprocessed about five-year data (October 2017 to September 2022) using the Real-Time Deforestation Detection System (DETER), construct recurrent deep learning-based prediction model consisting of two long short term memory (LSTM) layers and two fully-connected layers. In the case of the classification model, it has an additional softmax layer. You can evaluate the mesh-wise model performance by applying an incremental training and testing method for two-year data (October 2020 to September 2022).<br/>
  - __features__:
      1. binary of whether deforestation occurred or not
      2. area where deforestation occurred
      3. ditance to the closest road (unit: meter)<br />
  - __input sequence length__: 12 months<br />

- 
