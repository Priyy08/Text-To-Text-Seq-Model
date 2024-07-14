# Text Generation with LSTM

This project demonstrates a text generation model using LSTM neural networks in TensorFlow. The model is trained on a text dataset to predict the next word in a sequence.This model is trained on Data science Question answer pairs.

## Files

- `train.py`: Script for data preprocessing, model training, and text generation.
- `chat.py`: Script to directly generate text using pretrained model and tokenizer provided in repo.
- `DS.txt`: Training data file.
- `mytoken.pickle`: Tokenizer file.
- `text_generation_model6.h5`: Saved pre-trained model.
- `requirements.txt`: All necessary python imports to run python scripts.

## Setup

1. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the training script:**
    ```
    python train.py
    ```
    Note:Use train.py to train the model on my given dataset "DS.txt" to demonstrate how model is trained.You can also use your own Dataset to train similar models.
    After successfully running the script the model will be saved on your current directory using model.save() function and its tokenizer will also be saved using 
    pickle.

2. **Run the chat script:**
    ```
    python chat.py
    ```
    Note:chat.py script only used to chat with my given pretrained model and saved tokenizer. You can change the prompt by editing in "prompt" varaible in chat.py 
    script.
    
## Usage

1. **Running environments**
    -> To run these scripts preferably use google colab or jupyter notebook environments to avoid internal pc error for running the models.
   
2. **Training the Model:**
    -> To Train the Model only use train.py script by running "python train.py" command in terminal.
    
2. **Chatting with the Model:**
    -> To directly try the pretrained model and chat with it use "chat.py" script , use only those prompts which are related to dataset otherwise it will generate 
       unwanted responses.
