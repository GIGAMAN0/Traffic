 # Traffic Sign Classification: A Step-by-Step Guide

This repository contains a Python script (`traffic.py`) that can be used to train a convolutional neural network (CNN) model to classify traffic signs. The model is trained on a dataset of images of traffic signs, and can be used to identify the type of traffic sign in a given image.

## Step 1: Load the Data

The first step is to load the data into a format that can be used by the CNN model. The `load_data()` function in `traffic.py` does this by iterating over the directories in the `data_directory` and loading all of the images in each directory. The images are then resized to a consistent size and converted into numpy arrays. The labels for each image are also extracted from the directory names.

```python
def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_dir):
            img_path = os.path.join(category_dir, filename)
            
            # Read and resize image using OpenCV
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Append image and label to the lists
            images.append(img)
            labels.append(category)

    return images, labels
```

## Step 2: Split the Data into Training and Testing Sets

Once the data has been loaded, it needs to be split into training and testing sets. The `train_test_split()`
