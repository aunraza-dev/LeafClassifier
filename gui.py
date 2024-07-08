import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from tkinter import Tk, Label, Button, filedialog, StringVar
from PIL import ImageTk, Image

model = tf.keras.models.load_model('leaf_classifier_model.h5')

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        test_image = image.load_img(file_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        result = model.predict(test_image)
        predicted_class = np.argmax(result, axis=1)
        
        class_names = ['Disease Cotton Leaf', 'Disease Cotton Plant', 'Fresh Cotton Leaf', 'Fresh Cotton Plant'] 
        prediction_text.set(f'Prediction: {class_names[predicted_class[0]]}')

        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

root = Tk()
root.title("Leaf Classifier")
panel = Label(root)
panel.pack()

load_button = Button(root, text="Load Image", command=load_and_predict_image)
load_button.pack()

prediction_text = StringVar()
prediction_label = Label(root, textvariable=prediction_text)
prediction_label.pack()

root.mainloop()
