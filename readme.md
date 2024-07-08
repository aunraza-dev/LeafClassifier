# Leaf Classifier

A Convolutional Neural Network (CNN) based image classifier for classifying different types of leaves. This project includes training the model using TensorFlow and Keras, and a GUI built with `tkinter` for loading images and making predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [GUI Application](#gui-application)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/leaf-classifier.git
    cd leaf-classifier
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow numpy pillow
    ```

## Usage

### Model Training

1. Place your training, validation, and test images in the `data` directory with the following structure:
    ```
    data/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   ├── class3/
    │   └── class4/
    ├── test/
    │   ├── class1/
    │   ├── class2/
    │   ├── class3/
    │   └── class4/
    └── val/
        ├── class1/
        ├── class2/
        ├── class3/
        └── class4/
    ```

2. Run the `main.py` script to train and save the model:
    ```bash
    python main.py
    ```

### GUI Application

1. After training and saving the model, you can use the GUI to load images and get predictions:
    ```python
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing import image
    from tkinter import Tk, Label, Button, filedialog, StringVar
    from PIL import ImageTk, Image

    # Load the trained model
    model = tf.keras.models.load_model('leaf_classifier_model.h5')

    # Function to load and process the image
    def load_and_predict_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            test_image = image.load_img(file_path, target_size=(64, 64))
            test_image = image.img_to_array(test_image)
            test_image = test_image / 255.0
            test_image = np.expand_dims(test_image, axis=0)
            
            result = model.predict(test_image)
            predicted_class = np.argmax(result, axis=1)
            
            class_names = ['Class1', 'Class2', 'Class3', 'Class4']  # Replace with actual class names
            prediction_text.set(f'Prediction: {class_names[predicted_class[0]]}')

            img = Image.open(file_path)
            img = img.resize((250, 250))
            img = ImageTk.PhotoImage(img)
            panel.config(image=img)
            panel.image = img

    # Set up the GUI
    root = Tk()
    root.title("Leaf Classifier")

    # Create a label to display the selected image
    panel = Label(root)
    panel.pack()

    # Create a button to load an image
    load_button = Button(root, text="Load Image", command=load_and_predict_image)
    load_button.pack()

    # Create a label to display the prediction
    prediction_text = StringVar()
    prediction_label = Label(root, textvariable=prediction_text)
    prediction_label.pack()

    root.mainloop()
    ```

2. Run the above script to start the GUI.

## Model Training

The model is a simple CNN with the following architecture:
- Convolutional layer with 32 filters, kernel size of 3, ReLU activation
- Max pooling layer with pool size of 2
- Another convolutional layer with 32 filters, kernel size of 3, ReLU activation
- Another max pooling layer with pool size of 2
- Flattening layer
- Dense layer with 128 units and ReLU activation
- Output dense layer with 4 units (for 4 classes) and softmax activation

The model is compiled with the Adam optimizer and categorical cross-entropy loss.

## GUI Application

The GUI is built with `tkinter` and allows users to load an image and see the prediction made by the trained model. The selected image is displayed in the GUI along with the predicted class.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Pillow](https://python-pillow.org/)
- [tkinter](https://docs.python.org/3/library/tkinter.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
