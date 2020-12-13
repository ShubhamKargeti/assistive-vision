from __future__ import division, print_function
from cv2 import cv2
import numpy as np
from tqdm import tqdm
from gtts import gTTS
import sys
import os
import glob
import re
import numpy as np
from torchvision import transforms 
import torch
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename


transform_test = transforms.Compose([ 
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

data_loader = get_loader(transform=transform_test,    
                         mode='test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder_file = 'encoder-1.pkl'
decoder_file = 'decoder-1.pkl'
print('Loading encoder', encoder_file, 'and decoder', decoder_file)

embed_size = 512
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Load your trained model
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file),map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file),map_location=torch.device('cpu')))

encoder.to(device)
decoder.to(device)

def clean_sentence(output):
    clean_output = output[1:-1] 
    words = []
    for idx in clean_output:
        words.append(data_loader.dataset.vocab.idx2word[idx])
    sentence = ' '.join(words)
    return sentence


def get_prediction(img_path):
    PIL_image = Image.open(img_path).convert('RGB')
    image = transform_test(PIL_image).unsqueeze(0)
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    return sentence



app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def return_speech(text):
    speech = gTTS(text=text,lang='en',slow=False)
    speech.save('static/new.mp3')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab

    if request.method == 'POST':

	    img = request.files['file1']

	    img.save('static/file.jpg')

	    print("="*50)
	    print("IMAGE SAVED")

	    preds = get_prediction('static/file.jpg')
	    return_speech(preds)
    return render_template('predict.html', data=preds)


if __name__ == "__main__":
    app.run(debug=True)
