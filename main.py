import requests
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from datetime import datetime
import time


class CaptchaSolver:
    def __init__(self, model_path='captcha_model.h5', mappings_path='captcha_model_mappings.pkl'):
        self.model = None
        self.char_to_num = {}
        self.num_to_char = {}
        self.characters = ""
        self.img_width = 128
        self.img_height = 64
        self.max_length = 5

        if os.path.exists(model_path) and os.path.exists(mappings_path):
            self.model = keras.models.load_model(model_path)
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
            self.char_to_num = mappings['char_to_num']
            self.num_to_char = mappings['num_to_char']
            self.characters = mappings['characters']
            self.img_width = mappings['img_width']
            self.img_height = mappings['img_height']
            self.max_length = mappings['max_length']
            print("✓ Model loaded")
        else:
            raise FileNotFoundError("Trained model or mappings file not found.")

    def fetch_captcha(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, timeout=10)
        response.raise_for_status()
        save_path = f"captcha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return save_path

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(binary, (self.img_width, self.img_height))
        normalized = resized.astype('float32') / 255.0
        return normalized.reshape(1, self.img_height, self.img_width, 1)

    def predict(self, image_path):
        input_img = self.preprocess_image(image_path)
        preds = self.model.predict(input_img, verbose=0)
        result = ""
        for i in range(self.max_length):
            index = np.argmax(preds[i][0])
            result += self.num_to_char.get(index, '')
        return result


def main():
    url = "https://student.srmap.edu.in/srmapstudentcorner/captchas"
    solver = CaptchaSolver()
    image_path = solver.fetch_captcha(url)
    start_time = time.time()
    result = solver.predict(image_path)
    end_time = time.time()
    print(f"\n🧠 Predicted CAPTCHA: {result}")
    print(f"⏱️ Time taken: {end_time - start_time:.4f} seconds")
    print(f"💾 Image saved as: {image_path}")


if __name__ == "__main__":
    main()
