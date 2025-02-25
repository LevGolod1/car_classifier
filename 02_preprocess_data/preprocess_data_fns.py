import matplotlib.pyplot as plt

import torch
import clip
from PIL import Image
from PIL import ImageOps
import cv2

import numpy as np
import pandas as pd
import copy
import os
import glob

import datetime as dt
import time
from timethis import timethis
import subprocess
from common_params import parent_directory_images, parent_directory_url_csvs

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


categories = [
    'car exterior',
    'car interior',
    'car key',
    # 'dealership banner',
    'advertisement',
    # 'warranty',
    'dashboard',
    'wireless charger',
    'engine bay',
    'gauge cluster',
    'gear selector',
    'infotainment/navigation screen',
    'moonroof',
    'paperwork',
    'steering wheel',
    'wheels closeup'
]
categories = sorted(set(categories))

def create_image_files_df():
    '''
    File naming example '/Users/levgolod/Projects/car_classifier/data/autotrader/vehicle_images/make-bmw/model-3-series/vehicle_id-701932606/20f1aa5ef85b48f7ac676a1e82147582.jpg',

    :return:
    '''
    parent_directory_images
    image_files = sorted(glob.glob(parent_directory_images + '**/*.jpg', recursive=True))
    print(len(image_files))
    # image_files = image_files[:4]
    df = pd.DataFrame({'filepath': image_files})
    df['filename'] = df['filepath'].apply(lambda x: os.path.basename(x))
    df['make'] = df['filepath'].apply(lambda x: x.split('/')[8].replace('make-',''))
    df['model'] = df['filepath'].apply(lambda x: x.split('/')[9].replace('model-',''))
    df['vehicle_id'] = df['filepath'].apply(lambda x: x.split('/')[10].replace('vehicle_id-',''))
    return df


def get_predicted_categories_clip(image_path:str, categories:list)->dict:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(categories).to(device)

    # Predict category
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        probs = (image_features @ text_features.T).softmax(dim=-1)

    probs_w_categories = pd.Series(dict(zip(categories, np.array(probs)[0]))).sort_values(ascending=False)
    return probs_w_categories


def categorize_and_plot(image_path, categories):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display

    predicted_categories_all = get_predicted_categories_clip(image_path, categories)
    predictions = dict(predicted_categories_all.head(3))
    # image = cv2.resize(image, (500, 500))


    # Convert image to writable format
    overlay = image.copy()

    # Define text box properties
    x, y, w, h = 10, 10, 350, 80  # Top-right position
    alpha = 0.9  # Transparency level

    # Draw semi-transparent rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Black box
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Blend with original image

    # Add text on top of the rectangle
    y_offset = y + 20
    for label, prob in predictions.items():
        text = f"{label}: {int(prob*100):.0f}%"
        cv2.putText(image, text, (x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 20  # Line spacing

    # Display the image
    plt.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()
    return plt