
import os #operating system
import cv2 #opencv used for image processing
import numpy as np #used for sharpening the edges od the images
from tensorflow.keras.models import Sequential #used for sequential allotment of uploded images
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #used for converting label image to binary image

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) # image resizing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #grayscale conversion
    hist_eq_image = cv2.equalizeHist(gray_image) #histogram = contrast of the image for visibility
    high_pass_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) #sharpening ofthe image
    filtered_image = cv2.filter2D(hist_eq_image, -1, high_pass_filter)
    _, binary_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY) #image to binary
    return image, gray_image, binary_image

# Load dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    for label in ['no', 'yes', 'benign', 'malignant']:
        for img_path in os.listdir(os.path.join(dataset_path, label)):
            full_img_path = os.path.join(dataset_path, label, img_path)
            image = preprocess_image(full_img_path)[1]  # Use the gray_image for training
            images.append(img_to_array(image)) #append = adding value to last part of the array
            labels.append(label)
    images = np.array(images, dtype="float32") #float = variable with decimal value
    labels = np.array(labels)
    return images, labels

# Build CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), #conv2d = maintaining texture of image and extracting input image 
        MaxPooling2D((2, 2)), #maxpooling = reduces the spatial dimension(reduces complexiy)
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), 
        Flatten(), # flatten = 2D to 1D
        Dense(128, activation='relu'), # dense = compacting all layers
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess dataset
dataset_path = 'C:/Users/rrmcb/OneDrive/Desktop/cancer detection/brain tumor/data sets'
images, labels = load_dataset(dataset_path)

# Encode labels 
label_encoder = LabelEncoder() #label image to binary image
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Save label encoder classes
np.save('classes.npy', label_encoder.classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build and train model
input_shape = X_train.shape[1:]
model = build_model(input_shape)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save the model
model.save('brain_tumor_detection_model.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
