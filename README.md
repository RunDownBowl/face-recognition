# Facial Age Classification (CNN)

A convolutional neural network (CNN) that looks at a face image and predicts the person's age group: **YOUNG**, **MIDDLE** or **OLD**.

## Results

| Metric | Value |
|---|---|
| Test accuracy (1,991 held-out images) | **68.8%** |
| Random-guess baseline (3 classes) | 33.3% |

The model is about **2× better than random chance**. The main limit on accuracy is the dataset (small, low-resolution images and an unbalanced class split: more than half of the images are MIDDLE).

## Dataset

- 19,906 labelled face images (`train.csv` holds the image ID and the age class).
- Images are resized to 80 × 80 RGB and scaled to the range 0–1.
- 90% / 10% train–test split (`random_state=42`), with 10% of the training data kept for validation.
- The image files are too large for this repository, so only `train.csv` is included.

## Model

Built with TensorFlow / Keras:

- 2 × Conv2D (16 filters, 3 × 3) + MaxPooling
- Flatten → Dense 256 → Dense 256 → Dense 128 (ReLU, L2 regularisation)
- Dense 3 (softmax)
- Optimiser: Adam · Loss: categorical cross-entropy

## What I learned

- How to load, label and preprocess a large image dataset in Python.
- How to build and train a CNN in Keras and test it on unseen data.
- How to read results honestly: the model learns useful patterns, but better data (more images, higher resolution, balanced classes) is the next step to higher accuracy.

## How to run

1. Install: `pip install tensorflow pandas numpy pillow scikit-learn matplotlib`
2. Put the face images in `./faces/Train/` (file names must match the IDs in `train.csv`).
3. Open and run `face_age_rec.ipynb` in Jupyter or Google Colab.

## Tech

Python · TensorFlow · Keras · NumPy · Pandas · Pillow · scikit-learn
