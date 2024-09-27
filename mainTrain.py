import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from skimage import morphology, measure 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

# Constants input layer
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) #image resizing 128*128 pixel
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #grayscale conversion (RGB to Gray)
    hist_eq_image = cv2.equalizeHist(gray_image) #contrast the image for visibility
    high_pass_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) #sharpening the image 
    filtered_image = cv2.filter2D(hist_eq_image, -1, high_pass_filter)
    _, binary_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY) #binary thresholding = grayscale images to binary images
    return image, gray_image, binary_image

# Perform segmentation using region growing
def segment_image(binary_image): #comparing the tissue with neighbour tissue for detection
    labeled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)
    return labeled_image, num_labels

# Apply morphological operations
def morphological_operations(labeled_image): #morphological op = reduce noise,gaps
    dilated_image = morphology.dilation(labeled_image) #dilation = add pixels for image to make boundaries large, fill gaps and connect disjoints
    eroded_image = morphology.erosion(dilated_image) #erosion = remove pixels from image to reduce boundaries large and reduce noise
    return eroded_image

# mean intensity
def extract_mean(image): #mean intensity = brightness
    return np.mean(image)

# contrast
def extract_contrast(image): #variability of brightness in the tumor and non tumor regions
    return np.std(image)

#entropy
def extract_entropy(image): # maintain the complexity of the image texture
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    return entropy

#energy
def extract_energy(image): #sum of squared pixels for differentating b/w healty and tumor tissues
    return np.sum(image ** 2)

# Extract features from the image
def extract_features(image):
    mean = np.mean(image)
    contrast = np.std(image)
    entropy = -np.sum(image * np.log2(image + 1e-9))
    energy = np.sum(image ** 2)
    return [mean, contrast, entropy, energy]

# Load the model
model = load_model('brain_tumor_detection_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                return "No file uploaded."

            # Ensure 'uploads' directory exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            
            # Save the uploaded file
            file.save(file_path)
            
            # Preprocess the image
            image, gray_image, binary_image = preprocess_image(file_path)
            
            # Perform segmentation
            segmented_image, num_labels = segment_image(binary_image)
            
            # Apply morphological operations
            morph_image = morphological_operations(segmented_image)
            
            # Extract features
            features = extract_features(morph_image)
            
            # Predict using the trained model
            input_image = img_to_array(gray_image).reshape((1, IMG_HEIGHT, IMG_WIDTH, 1))
            prediction = model.predict(input_image)
            predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0] # argmax = find indices of the maximum values 
            # along specified axis in an array
            
            # Display results
            return render_template('result.html', 
                                   image_path=file.filename, 
                                   label=predicted_label, 
                                   features=features)
        
        except FileNotFoundError:
            return "FileNotFoundError: File not found."
        
        except PermissionError:
            return "PermissionError: Permission denied."
        
        except Exception as e:
            return f"Error processing file: {e}"
    
    return '''
    <!doctype html>
    <title>Brain Tumor Detection</title>
    <h1>Upload MRI Image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
