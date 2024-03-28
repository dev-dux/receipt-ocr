from flask import request, jsonify
from app import app
from ultralytics import YOLO
import json
from datetime import datetime
from PIL import Image
import pytesseract
import os
import tensorflow as tf
import tensorflow_hub as hub
import logging
import uuid
logging.basicConfig(level=logging.INFO)

# Your OCR logic here
def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)

def image_enhancement(image_path, enhanced_image_path):
    logging.info("Started image enhancement...")
    saved_model_path = 'app/archive'
    hr_image = preprocess_image(image_path)
    model = hub.load(saved_model_path)
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    save_image(tf.squeeze(fake_image), filename=enhanced_image_path)
    logging.info("Completed image enhancement!\n")

def predict(image_path):
    model_path = 'app/best.pt'
    logging.info("Loading model...")
    model = YOLO(model_path)
    logging.info(f"Loaded model: {model._get_name()}!")
    logging.info("Predicting image labellings...")
    results = model([image_path])
    logging.info("Generated labelled images!")
    for result in results:
        # timestamp = int(round(datetime.now().timestamp()))
        result_path = f'app/data/save_dir/cls_{uuid.uuid4().__str__}'
        result.save_crop(result_path)
        logging.info(f"Labelled images stored at: {os.path.abspath(result_path)}")
        return result_path

def ocr(image_dir):
    data = {
        'store_name': '',
        'store_address': '',
        'transaction_date': '',
        'transaction_time': '',
        'invoice_number': '',
        'items': []
    }
    logging.info("Reading labelled images...")
    for cls in os.listdir(image_dir):
        cls_path = os.path.join(image_dir, cls)
        for image in os.listdir(cls_path):
            image_path = os.path.join(cls_path, image)
            if os.path.isfile(image_path):
                logging.info(f"\n============= On image {cls} =============")
                img = Image.open(image_path)
                logging.info("Extracting text...")
                text = pytesseract.image_to_string(img).rstrip().replace("\n", " ")
                logging.info("Completed text extraction!")
                if cls != 'items':
                    if cls == 'product':
                        data['items'].append({'product_name': text, 'product_quantity': None})
                    else:
                        data[cls] = text
    # json_object = json.dumps(data, indent=4)
    # result_json_file_path = "ocr.json"
    # with open(result_json_file_path, "w") as outfile:
    #     outfile.write(json_object)
    # logging.info(f"\nResult stored at: {os.path.abspath(result_json_file_path)}")
    return data
    