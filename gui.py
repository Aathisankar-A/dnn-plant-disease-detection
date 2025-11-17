import tkinter as tk
from tkinter import filedialog
from tkinter import Label
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os

IMG_SIZE = 128

# LOAD MODEL
model = tf.keras.models.load_model("plant_disease_manual.h5")

# LOAD CLASS NAMES FROM FOLDER
data_dir = "plantVillage"
class_names = sorted(os.listdir(data_dir))

# GUI SETUP
root = tk.Tk()
root.title("Plant Disease Detection")
root.geometry("600x700")

def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    img = Image.open(file_path)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0

    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    class_index = np.argmax(prediction)

    disease_name = class_names[class_index]  # Convert index → disease name

    # SHOW IMAGE
    img_tk = ImageTk.PhotoImage(img.resize((300, 300)))
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # SHOW PREDICTION
    result_label.config(
        text=f"Prediction:\n{disease_name.replace('__', ' → ')}",
        font=("Arial", 18),
        fg="green"
    )

# GUI ELEMENTS
title_label = Label(root, text="Upload a leaf image", font=("Arial", 24))
title_label.pack(pady=20)

upload_btn = tk.Button(root, text="Upload Image", font=("Arial", 16), command=upload_image)
upload_btn.pack(pady=20)

image_label = Label(root)
image_label.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
