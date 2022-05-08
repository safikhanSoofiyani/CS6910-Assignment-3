# CS6910-Assignment-3 : Recurrent Neural network to build a transliteration system
Assignment 3 submission for the course CS6910 Fundamentals of Deep Learning. <br>
Check this link for the task description: [Assignment link](https://wandb.ai/miteshk/assignments/reports/Assignment-3--Vmlldzo0NjQwMDc)


Team Members : **Vamsi Sai Krishna Malineni (OE20S302)**, **Mohammed Safi Ur Rahman Khan (CS21M035)** 
---
## Overview :
The purpose of this assignment were :
1. Building a transliteration system using Recurrent Neural Networks.
2. Comparing different cells such as vanilla RNN, LSTM and GRU. 
3. Implementing attention mechanism and understand how these overcome the limitations of vanilla seq2seq models.
4. Fine tuning GPT2 transformer model to generate lyrics based on the prompt given.

The link to the wandb report: 

---
## General Instructions:
1. Install the required libraries using the following command :

```python 
pip install -r requirements.txt
```
2. The solution to the assignment is presented in the following notebooks :
    1. `RNN.ipynb` : This notebook corresponds to training an rnn without attention
    2. `RNN_with_Attention.ipynb` : This notebook corresponds to training an rnn with attention, visualizing the connectivity and plotting the attention maps
    3. `Lyrics_Generation.ipynb` : This notebook corresponds to training GPT2 transformer model for lyrics generation
3.  If you are running the jupyter notebooks on colab, the libraries from the `requirements.txt` file are preinstalled, with the `exception` of the following:
    * `wandb`
    * `transformers`
    *  `datasets`
    <br/> <br/> You can install wandb by using the following command :
```python
!pip install wandb
!pip install transformers
!pip install datasets

```
4. The dataset for the RNN part of the assignment can be found at : [RNN Dataset Link](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar)
5. The dataset for the Transformers part of the assignment can be found at : [Transformer Dataset Link](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres)
---
## Building and training a RNN model :
You can run the `RNN.ipynb` notebook sequentially ( cell by cell ) to train the RNN model.

#### 1. Getting training data :
The training and validation data in encoded form is obtained using the following code snippet:
```python
encoder_input_data, decoder_input_data, decoder_target_data, encoder_val_input_data, decoder_val_input_data, decoder_val_target_data = one_hot_encoding(input, target, val_input, val_target, input_tokens, target_tokens)
```
where :
1. `input` : array of words of the input language (i.e. English) present in training dataset
2. `target` : array of words of the target language (i.e. Telugu) present in training dataset
3. `val_input` : array of words of the input language (i.e. English) present in validation dataset
4. `val_target` : array of words of the target language (i.e. Telugu) present in validation dataset
5. `input_tokens` : list of charaters in the input language
6. `target_tokens` : list of characters in the target language

#### 2. Building Model :
The following function is used to build the RNN model:
```python
build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size)
```
where :
1.  `num_encoders` : number of encoder layers
2.  `num_decoders` : number of decoder layers
3.  `cell` : type of the rnn cell ( simpleRNN, LSTM ,GRU )
4.  `embed_size` : size of the embedding 
5.  `dropout` : % of dropout (in decimals)
6.  `hidden_layer_size` : dimensionalilty of output space

#### 3. Hyperparameter sweeps :
Use the  following function to perform hyperparameter sweeps in wandb:
```python 
sweeper(project_name,entity_name)
```
where:
1. `project_name` : Enter the wandb project name
2. `entity_name` : Enter the wandb entity name

The various hyperparameters used are : 
```python 
hyperparameters = {
          "cell":   {"values":  ["RNN","GRU","LSTM"]},
          "embed_size": {"values":   [16,32,64,256]},
          "hidden_layer_size":  {"values":  [16,32,64,256]},
          "num_layers": {"values":   [1,2,3]},
          "dropout":    {"values":  [0.2,0.3,0.4]},
          "epochs": {"values":   [5,10,15,20]},
          "batch_size": {"values":  [32,64]},
          "optimizer":  {"values":    ["adam", "rmsprop", "nadam"]}
      }
```

#### 4. Getting Test data :
The test data in encoded form is obtained by using the following code snipppet :
```python 
encoder_test_input_data, decoder_test_input_data, decoder_test_target_data = one_hot_encoding_test(test_input, test_target, input_tokens, target_tokens)
```
where :
1. `test_input` : array of words of the input language (i.e. English) present in test dataset
2. `test_target` : array of words of the target language (i.e. Telugu) present in test dataset
3. `input_tokens` : list of charaters in the input language
4. `target_tokens` : list of characters in the target language


#### 5. Testing the best performing model :
The best performing model from the hyperparameter sweeps is fed with test data. Use the following code snippet to find the test accuracy :
```python
testing()
```
---
## Building and training a RNN model with attention mechanism:
You can run the `RNN_with_Attention.ipynb` notebook sequentially ( cell by cell ) to train the RNN model.

#### 1. Getting training data :
The training and validation data in encoded form is obtained using the following code snippet:
```python
encoder_input_data, decoder_input_data, decoder_target_data, encoder_val_input_data, decoder_val_input_data, decoder_val_target_data = one_hot_encoding(input, target, val_input, val_target, input_tokens, target_tokens)
```
where :
1. `input` : array of words of the input language (i.e. English) present in training dataset
2. `target` : array of words of the target language (i.e. Telugu) present in training dataset
3. `val_input` : array of words of the input language (i.e. English) present in validation dataset
4. `val_target` : array of words of the target language (i.e. Telugu) present in validation dataset
5. `input_tokens` : list of charaters in the input language
6. `target_tokens` : list of characters in the target language

#### 2. Building Model :
The following function is used to build the RNN model:
```python
build_model(num_encoders, num_decoders, cell, embed_size, dropout, hidden_layer_size)
```
where :
1.  `num_encoders` : number of encoder layers
2.  `num_decoders` : number of decoder layers
3.  `cell` : type of the rnn cell ( simpleRNN, LSTM ,GRU )
4.  `embed_size` : size of the embedding 
5.  `dropout` : % of dropout (in decimals)
6.  `hidden_layer_size` : dimensionalilty of output space

#### 3. Hyperparameter sweeps :
Use the  following function to perform hyperparameter sweeps in wandb:
```python 
sweeper(project_name,entity_name)
```
where:
1. `project_name` : Enter the wandb project name
2. `entity_name` : Enter the wandb entity name

The various hyperparameters used are : 
```python 
hyperparameters = {
          "cell":{ "values":["RNN","GRU","LSTM"]},
          "embed_size":{ "values":[16,32,64,256]},
          "hidden_layer_size":{"values":[16,32,64,256]},
          "num_layers":{"values":[1,2,3]},
          "dropout":{"values":[0.2,0.3,0.4]},
          "epochs":{"values":[5,10,15,20]},
          "batch_size":{"values":[32,64]},
          "optimizer":{"values":["adam", "rmsprop", "nadam"]}
      }
```

#### 4. Getting Test data :
The test data in encoded form is obtained by using the following code snipppet :
```python 
encoder_test_input_data, decoder_test_input_data, decoder_test_target_data = one_hot_encoding_test(test_input, test_target, input_tokens, target_tokens)
```
where :
1. `test_input` : array of words of the input language (i.e. English) present in test dataset
2. `test_target` : array of words of the target language (i.e. Telugu) present in test dataset
3. `input_tokens` : list of charaters in the input language
4. `target_tokens` : list of characters in the target language


#### 5. Testing the best performing model :
The best performing model from the hyperparameter sweeps is fed with test data. Use the following code snippet to find the test accuracy :
```python
testing()
```
#### 6. Plotting Attention maps :
The attention maps for the model can be built running the following section of ipynb notebook : 
```python
Plotting attention maps
```
#### 7. Visualising Connectivity  : 
The connectivity can be visualized using either of the following functions :

```python
visualize_gif(index)
```
```python 
visualize(index)
```
where :
1. `visualize_gif(index)` returns a gif format of connectivity visualization of the word at the given `index` 
2. `visulaize(index)` returns an image format of connectivity visualization of the word at the given `index`

The resources used are :
1. https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff
2. https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
3.  https://keras.io/examples/nlp/lstm_seq2seq/

---
## Training and fine tuning  GPT2 Transformers:
<br/> The dataset used for fine tuning GPT2 model can be found here : https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres 
<br/> This dataset contains lyrics from 6 muscial genres (data is scraped from the website vagalume.com.br)
<br/> The notebook is split into these following segments :
1. `Library imports` : This section imports the required libraries for the task.
2. `Data Preperation`: This section builds the train,test and validation datasets.
3. `Model Training` : This section deals with the model training by running `run_clm.py` file.
4. `Lyrics Generation` : This section generates lyrics by running `run_generation.py` file. The lyrics generation is based on the prompt given : " I love deep learning ".

The resources used for this question are as follows:
1. https://github.com/sfs0126/Lyric-Generator-fine-tuned-GPT-2
2. https://github.com/huggingface/transformers/tree/master/examples/pytorch
3. https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7

