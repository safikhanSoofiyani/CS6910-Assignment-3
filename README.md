# CS6910-Assignment-3 : Recurrent Neural network to build a transliteration system
Assignment 3 submission for the course CS6910 Fundamentals of Deep Learning. <br>
Check this link for the task description: [Assignment link](https://wandb.ai/miteshk/assignments/reports/Assignment-3--Vmlldzo0NjQwMDc)


Team Members : **Vamsi Sai Krishna Malineni (OE20S302)**, **Mohammed Safi Ur Rahman Khan (CS21M035)** 

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

