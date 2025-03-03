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
import pytesseract

import datetime as dt
import time
from timethis import timethis
import subprocess
from common_params import parent_directory_images, parent_directory_url_csvs

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


body_style_dict = {'sedan': 0, 'sportscar': 1, 'suv': 2, 'truck': 3, 'van': 4, 'wagon': 5}

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


def preprocess_image_for_banner_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def banner_text_box(image_path, preprocessing_fn, plot=False, margin=25):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height = image.shape[0]

    # Apply thresholding to enhance text visibility
    image_processed = preprocessing_fn(image)

    # Detect text regions using Tesseract
    custom_config = r'--oem 3 --psm 3'  # OCR engine mode and page segmentation mode
    data = pytesseract.image_to_data(image_processed, config=custom_config, output_type=pytesseract.Output.DICT)

    # Draw bounding boxes around detected text
    data['is_banner']=[0] * len(data['text'])
    data['top_bottom']=[None] * len(data['text'])
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 50:  # Confidence threshold
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            height_pct = int(100*y/image_height)
            depth_pct = int(100*h/image_height)
            is_banner = all([
                depth_pct <= 20,
                (height_pct <= margin or height_pct >= (100-margin))
            ])
            if is_banner:
                data['is_banner'][i] = 1 #
                data['top_bottom'][i] = 'top' if height_pct <= margin else 'bottom' if height_pct >= (100-margin) else None
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if plot:
        plt.imshow(image)
        plt.axis("off")  # Hide axis
        plt.show()

    return data


def crop_out_text(image_path: str, new_path: str = None, plot: bool = False, margin: int = 20) -> dict:
    text_banner_data = banner_text_box(image_path, preprocess_image_for_banner_text, plot=False, margin=margin)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height = image.shape[0]

    banner_indices_top = [i for i, x in enumerate(text_banner_data['top_bottom']) if x == 'top']
    if banner_indices_top:

        text_banner_top = {}
        for k, v in text_banner_data.items():
            text_banner_top[k] = [x for i, x in enumerate(v) if i in banner_indices_top]

        argmin_top = np.argmax(text_banner_top['top'])
        top_slice = text_banner_top['top'][argmin_top] + text_banner_top['height'][argmin_top]
    else:
        top_slice = 0

    banner_indices_bottom = [i for i, x in enumerate(text_banner_data['top_bottom']) if x == 'bottom']
    if banner_indices_bottom:

        text_banner_bottom = {}
        for k, v in text_banner_data.items():
            text_banner_bottom[k] = [x for i, x in enumerate(v) if i in banner_indices_bottom]

        bottom_slice = np.min(text_banner_bottom['top'])
    else:
        bottom_slice = image_height

    no_text_banner = all([
        len(banner_indices_top) == 0,
        len(banner_indices_bottom) == 0,
    ])
    if no_text_banner:
        # print('no_text_banner')
        cropped_image = probs_before = probs_after = probs_diff = None


    else:
        cropped_image = image[top_slice: bottom_slice, :, :]
        cv2.imwrite('cropped_image.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

        probs_before = get_predicted_categories_clip(
            image_path=image_path,
            categories=categories
        )['car exterior']

        probs_after = get_predicted_categories_clip(
            image_path='cropped_image.jpg',
            categories=categories
        )['car exterior']

        probs_diff = probs_after - probs_before

        if plot:
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
            message = f"prob of car exterior went from {probs_before:.1%} to {probs_after:.1%} after crop"

            plt.imshow(image)
            plt.axis("off")
            plt.title(message)

            # Show second image
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
            plt.imshow(cropped_image)
            plt.axis("off")
            plt.show()

    return {
        'cropped_image': cropped_image,
        'probs_before': probs_before,
        'probs_after': probs_after,
        'probs_diff': probs_diff
    }



def process_image_files(input_files:list, image_size:int=256) -> list:
    '''
    read image files and create np.array
    :param input_files: list where each element is the file path and the label
    :return: list where each element is the image array and the label (numeric)
    '''
    reduced_image_size = image_size/ 4.0
    if reduced_image_size != int(reduced_image_size):
        print(f'image_size must be cleanly divisible by 4')
        exit()

    results=[]
    for i,x in enumerate(input_files):
        # if i % 1000 ==0:
        #     print(i, dt.datetime.now())
        label = x[1]
        image_path=x[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image= cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array = np.array(gray)
        tensor = torch.tensor(image_array, dtype=torch.float32)
        # tensor = tensor.unsqueeze(0)
        results+=[(tensor,label)]
        # results+=[(image_array,label)]
    #
    return results

