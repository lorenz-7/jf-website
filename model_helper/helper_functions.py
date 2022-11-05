from keras.models import load_model
import numpy as np
import cv2


def test(model, data, custom_objects=None):
    model = load_model(model, custom_objects=custom_objects)
    image = np.expand_dims(data, axis=0)
    prediction = model.predict(image)
    return prediction


def return_color(color_map, data):
    data = data.astype(np.uint8)
    out = cv2.applyColorMap(data, color_map)
    return out


def read(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return lines


def load(model_name, custom_objects=None):
    model = load_model(model_name, custom_objects=custom_objects)
    return model


def predict(img, model, type):
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    if type == "test":
        img = img[:, :, :, 0]
    mask = model.predict(img)
    mask = np.squeeze(mask)
    #   mask = np.round(mask)
    #   mask = mask * 255
    return mask





def make_gray(img, channels):
    img = np.array(img)
    if channels == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif channels == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def colorize_mask(mask, colormap, channels):
    mask *= 255
    mask = mask.astype(np.uint8)
    mask = cv2.applyColorMap(mask, colormap)
    mask = cv2.cvtColor(mask, channels)
    #   mask[mask < 255] = 0
    return mask
