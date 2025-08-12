# Disaster_Prediction_Model

## Overview
This project is a machine learning-based image classification system that identifies the type of disaster depicted in a photo. The model is trained to recognize various disaster types such as earthquakes, wildfires, floods, hurricanes, and more.

The system processes input images and outputs a classification label corresponding to the disaster type. This solution can be used in disaster management systems to quickly identify the type of catastrophe, aiding in more efficient response efforts.

## Features
- Classifies images into different disaster types (e.g., earthquake, wildfire, flood, hurricane).
- Built using a Convolutional Neural Network (CNN) with TensorFlow and Keras.
- Achieves high accuracy in detecting disaster types from images.
- Easy to use command-line interface for testing new images.
  
## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/username/disaster-image-classifier.git
    cd disaster-image-classifier
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset** (if needed):
    - This project uses a dataset of labeled disaster images. You can download a suitable dataset like [Disaster Images Dataset](https://www.kaggle.com/datasets) and place it in the `data/` folder.

5. **Train the model** (optional, if not using a pre-trained model):
    ```bash
    python train.py
    ```

## Usage

### Predicting the Disaster Type for an Image

1. **Run the prediction script**:
    ```bash
    python predict.py --image path_to_image.jpg
    ```

2. **Example**:
    ```bash
    python predict.py --image samples/earthquake.jpg
    ```

3. The result will display the predicted disaster type, for example:
    ```
    Predicted Disaster: Earthquake
    ```

### Using Pre-trained Model

If you are using the pre-trained model included in the repository, simply skip the training step and run the `predict.py` script as shown above.


## Model Architecture

The disaster classifier is built using a Convolutional Neural Network (CNN) with the following architecture:

- **Input Layer**: Takes in images of size (224x224x3).
- **Convolutional Layers**: Multiple layers for feature extraction.
- **Pooling Layers**: MaxPooling to reduce spatial dimensions.
- **Fully Connected Layers**: Dense layers to classify disaster types.
- **Output Layer**: Softmax layer for multi-class classification.

## Example Disaster Types
The model is trained to classify the following disaster types:
- Earthquake
- Wildfire
- Flood
- Hurricane
- Tornado
- Tsunami

## Future Improvements
- Improve the accuracy of disaster detection by using a larger dataset.
- Add more disaster types for classification.
- Optimize the model for faster inference on edge devices.
- Build a web interface to upload and classify images directly from a browser.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any major changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Kaggle Disaster Dataset](https://www.kaggle.com/datasets) for providing labeled disaster images.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning framework.
