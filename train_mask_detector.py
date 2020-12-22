from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os

INITIAL_LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.20
MODEL = "models/mask_detector.model"

DATASET_FOLDER = "dataset"

DIR_NAME = os.path.dirname(__file__)
# Compute relative path to dataset folder
DIRECTORY = os.path.relpath(DATASET_FOLDER, DIR_NAME)

# Categories must match the dataset folder structure.
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

# Check if directory exists
if not os.path.exists(DIRECTORY):
	print(f"[INFO] {DIRECTORY} does not exist")
	exit(1)

print("[INFO] Loading images...")

# Loop over the categories and add info to data and labels list
for category in CATEGORIES:
	category_path = os.path.join(DIRECTORY, category)

	if not os.path.exists(category_path):
		print(f"[INFO] Category '{category}' does not exist. Skipping category")
		continue

	print(f"[INFO] Loading images for category: '{category}'")
	image_paths = list(paths.list_images(category_path))

	for image_path in image_paths:
		# Load image and resize to (244, 244)
		image = load_img(image_path, target_size=(224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)

		# Add data to data and labels lists
		data.append(image)
		labels.append(category)

if not data:
	print("[INFO] No images found")
	exit(1)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Partition the data into training and testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=TEST_SIZE, stratify=labels, random_state=42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CATEGORIES), activation="softmax")(headModel)

# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] Training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	steps_per_epoch=len(trainX) // BATCH_SIZE,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BATCH_SIZE,
	epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO] Evaluating model...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Serialize the model to disk
print("[INFO] Saving mask detector model...")
model.save(MODEL, save_format="h5")
