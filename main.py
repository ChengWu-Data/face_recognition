"""
Face Detection and Face Recognition Project

This script completes two tasks:

1. Face Detection
   - Uses the provided haarcascade_frontalface_default.xml file
   - Detects faces from the original Ariel Sharon images
   - Crops the detected faces
   - Saves them into a manually created folder named:
     "ariel sharon by hands"

2. Face Recognition
   - Loads six face classes
   - Splits the dataset into train and test sets with:
     test_size=0.3 and random_state=42
   - Uses Eigenfaces (PCA + classifier) for recognition
   - Prints the final accuracy

Important:
- This .py file is designed to stay in the same directory level as the dataset folder.
- Do not use absolute paths.
"""

# Import the built-in os module for working with folders and files.
import os

# Import the built-in re module for handling filename patterns.
import re

# Import OpenCV for image reading, grayscale conversion, face detection, and image writing.
import cv2

# Import NumPy for numerical operations on image arrays.
import numpy as np

# Import PCA from scikit-learn to implement the Eigenfaces method.
from sklearn.decomposition import PCA

# Import train_test_split to divide the data into training and testing sets.
from sklearn.model_selection import train_test_split

# Import StandardScaler to standardize pixel features before PCA.
from sklearn.preprocessing import StandardScaler

# Import KNeighborsClassifier as the classifier used after Eigenfaces feature extraction.
from sklearn.neighbors import KNeighborsClassifier

# Import accuracy_score to calculate the final recognition accuracy.
from sklearn.metrics import accuracy_score


# Define the expected image size after resizing all cropped faces.
IMAGE_SIZE = (100, 100)

# Define the dataset root folder name.
DATASET_FOLDER = "Faces"

# Define the required manual folder name for Ariel Sharon cropped faces.
ARIEL_OUTPUT_FOLDER = "ariel sharon by hands"

# Define the required six class labels for recognition.
REQUIRED_CLASSES = [
    "ariel_sharon",
    "chris evans",
    "chris hemsworth",
    "mark ruffalo",
    "robert_downey_jr",
    "scarlett_johansson"
]


def normalize_text(text):
    """
    Normalize a folder name or label so that different styles can still be matched.

    Examples:
    - "Ariel Sharon" -> "ariel sharon"
    - "ariel_sharon" -> "ariel sharon"
    - "Robert-Downey-Jr" -> "robert downey jr"

    Parameters:
        text (str): The original folder name or label.

    Returns:
        str: A cleaned and comparable version of the text.
    """

    # Convert the text to lowercase so matching becomes case-insensitive.
    text = text.lower()

    # Replace underscores and hyphens with spaces for easier comparison.
    text = text.replace("_", " ").replace("-", " ")

    # Collapse repeated spaces into one single space.
    text = re.sub(r"\s+", " ", text).strip()

    # Return the cleaned text.
    return text


def ensure_dataset_exists(dataset_path):
    """
    Check whether the dataset folder exists before any processing begins.

    Parameters:
        dataset_path (str): Relative path to the dataset folder.

    Raises:
        FileNotFoundError: If the dataset folder does not exist.
    """

    # If the dataset folder does not exist, stop the program and show a clear error.
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_path}' was not found. "
            f"Please place this script in the same level as the '{DATASET_FOLDER}' folder."
        )


def ensure_cascade_exists(cascade_path):
    """
    Check whether the Haar cascade XML file exists before face detection starts.

    Parameters:
        cascade_path (str): Relative path to the Haar cascade XML file.

    Raises:
        FileNotFoundError: If the XML file does not exist.
    """

    # If the cascade XML file is missing, stop the program and show a clear error.
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(
            f"Cascade file '{cascade_path}' was not found. "
            f"Please keep 'haarcascade_frontalface_default.xml' in the same folder as this script."
        )


def list_subfolders(folder_path):
    """
    Get all subfolder names inside a folder.

    Parameters:
        folder_path (str): Path to the parent folder.

    Returns:
        list[str]: A list of subfolder names.
    """

    # Create an empty list to store discovered subfolder names.
    subfolders = []

    # Loop through every item inside the folder.
    for item_name in os.listdir(folder_path):
        # Build the full path for the current item.
        item_path = os.path.join(folder_path, item_name)

        # Keep the item only if it is a folder.
        if os.path.isdir(item_path):
            subfolders.append(item_name)

    # Return the list of subfolder names.
    return subfolders


def find_matching_folder(dataset_path, target_name):
    """
    Find a folder whose normalized name matches the target label.

    Parameters:
        dataset_path (str): Path to the dataset folder.
        target_name (str): The label we want to find.

    Returns:
        str or None: The real folder name if found, otherwise None.
    """

    # Normalize the target label for flexible matching.
    normalized_target = normalize_text(target_name)

    # Get all subfolders in the dataset directory.
    all_subfolders = list_subfolders(dataset_path)

    # Loop through every discovered subfolder.
    for folder_name in all_subfolders:
        # Normalize the current folder name.
        normalized_folder = normalize_text(folder_name)

        # Return the real folder name if it matches the target.
        if normalized_folder == normalized_target:
            return folder_name

    # Return None if no folder matches.
    return None


def get_image_files(folder_path):
    """
    Collect valid image filenames from a folder.

    Parameters:
        folder_path (str): Path to an image folder.

    Returns:
        list[str]: Sorted image filenames.
    """

    # Define the image file extensions allowed in this project.
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    # Create an empty list to store valid image filenames.
    image_files = []

    # Loop through every file in the folder.
    for file_name in os.listdir(folder_path):
        # Keep the file only if its extension is one of the supported image formats.
        if file_name.lower().endswith(valid_extensions):
            image_files.append(file_name)

    # Sort the filenames so processing order stays stable.
    image_files.sort()

    # Return the sorted image list.
    return image_files


def infer_filename_pattern(reference_folder_path, default_prefix="image"):
    """
    Infer a saving pattern from an existing class folder so the Ariel Sharon cropped
    filenames can follow the same style as the other classes.

    This function tries to detect:
    - file extension
    - numeric width
    - prefix style

    Parameters:
        reference_folder_path (str): Path to an existing class folder used as style reference.
        default_prefix (str): Fallback prefix if the folder is empty or pattern is unclear.

    Returns:
        tuple[str, int, str]:
            - extension (for example ".jpg")
            - zero padding width (for example 4)
            - prefix (for example "chris_evans")
    """

    # Get all image filenames from the reference folder.
    reference_files = get_image_files(reference_folder_path)

    # Use a safe default extension.
    extension = ".jpg"

    # Use a safe default number width.
    number_width = 3

    # Use the provided fallback prefix first.
    prefix = default_prefix

    # If the reference folder has at least one image, inspect the first filename.
    if len(reference_files) > 0:
        # Read the first filename as the sample.
        sample_name = reference_files[0]

        # Split the filename into the name part and extension part.
        name_part, extension = os.path.splitext(sample_name)

        # Match names that end with digits, such as "chris_evans_001".
        match = re.match(r"^(.*?)(\d+)$", name_part)

        # If the pattern contains trailing digits, capture the prefix and digit width.
        if match:
            prefix = match.group(1)
            number_width = len(match.group(2))
        else:
            # If the name does not end with digits, reuse the full name and keep default width.
            prefix = name_part + "_"

    # Return the inferred extension, width, and prefix.
    return extension, number_width, prefix


def build_filename(prefix, index, number_width, extension):
    """
    Build a filename using the inferred naming pattern.

    Parameters:
        prefix (str): The text part before the number.
        index (int): The numeric image index.
        number_width (int): The zero-padding width.
        extension (str): The image extension.

    Returns:
        str: The final filename.
    """

    # Format the index with leading zeros according to the required width.
    number_text = str(index).zfill(number_width)

    # Join the prefix, padded number, and extension.
    return f"{prefix}{number_text}{extension}"


def detect_and_crop_faces(source_folder_path, output_folder_path, cascade_path, reference_folder_path):
    """
    Detect faces in the Ariel Sharon original images, crop the largest detected face,
    resize it, and save it into the required output folder.

    Parameters:
        source_folder_path (str): Folder containing Ariel Sharon original images.
        output_folder_path (str): Folder where cropped Ariel Sharon faces will be saved.
        cascade_path (str): Path to the provided Haar cascade XML file.
        reference_folder_path (str): Existing class folder used to copy filename style.

    Returns:
        int: Number of successfully saved cropped face images.
    """

    # Make sure the output folder exists before saving images.
    os.makedirs(output_folder_path, exist_ok=True)

    # Load the Haar cascade face detector from the provided XML file.
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read all valid image files from the Ariel Sharon source folder.
    image_files = get_image_files(source_folder_path)

    # Infer the filename style from one of the other class folders.
    extension, number_width, _ = infer_filename_pattern(reference_folder_path, default_prefix="sample_")

    # Use a prefix that follows the Ariel Sharon label in a clean way.
    save_prefix = "ariel_sharon_"

    # Create a counter for saved cropped face images.
    saved_count = 0

    # Create a counter for naming the output files.
    output_index = 1

    # Loop through every source image.
    for image_file in image_files:
        # Build the full path to the current image.
        image_path = os.path.join(source_folder_path, image_file)

        # Read the current image with OpenCV.
        image = cv2.imread(image_path)

        # Skip the image if OpenCV fails to read it.
        if image is None:
            continue

        # Convert the color image to grayscale because Haar cascade detection uses grayscale input.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current image.
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Skip the image if no face is detected.
        if len(faces) == 0:
            continue

        # Choose the largest detected face because it is usually the main subject.
        x, y, w, h = max(faces, key=lambda rectangle: rectangle[2] * rectangle[3])

        # Crop the face area from the grayscale image.
        face_crop = gray_image[y:y + h, x:x + w]

        # Resize the cropped face so all training samples have the same shape.
        face_crop = cv2.resize(face_crop, IMAGE_SIZE)

        # Build the output filename using the required numbering style.
        output_file_name = build_filename(save_prefix, output_index, number_width, extension)

        # Build the full path to the output file.
        output_file_path = os.path.join(output_folder_path, output_file_name)

        # Save the cropped face image to the output folder.
        cv2.imwrite(output_file_path, face_crop)

        # Increase the number of saved images.
        saved_count += 1

        # Move to the next output filename number.
        output_index += 1

    # Return the total number of cropped faces that were saved.
    return saved_count


def load_face_dataset(dataset_path, class_folder_mapping):
    """
    Load all face images from the six required classes.

    Each image is:
    - read in grayscale
    - resized to a fixed size
    - flattened into a 1D feature vector

    Parameters:
        dataset_path (str): Path to the full dataset folder.
        class_folder_mapping (dict): Maps class labels to real folder names.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X: image feature matrix
            - y: label array
    """

    # Create an empty list to store flattened image arrays.
    X = []

    # Create an empty list to store the corresponding labels.
    y = []

    # Loop through every required class label in fixed order.
    for class_label in REQUIRED_CLASSES:
        # Read the real folder name for the current class.
        folder_name = class_folder_mapping[class_label]

        # Build the full path to the current class folder.
        folder_path = os.path.join(dataset_path, folder_name)

        # Get all valid image files inside the current class folder.
        image_files = get_image_files(folder_path)

        # Loop through every image file in the current class folder.
        for image_file in image_files:
            # Build the full path to the current image file.
            image_path = os.path.join(folder_path, image_file)

            # Read the image directly in grayscale mode.
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Skip unreadable image files.
            if image is None:
                continue

            # Resize the face image to the common size used by the whole dataset.
            image = cv2.resize(image, IMAGE_SIZE)

            # Flatten the 2D image into a 1D vector for machine learning.
            image_vector = image.flatten()

            # Store the flattened image vector.
            X.append(image_vector)

            # Store the corresponding class label.
            y.append(class_label)

    # Convert the image list into a NumPy feature matrix.
    X = np.array(X)

    # Convert the label list into a NumPy array.
    y = np.array(y)

    # Return the feature matrix and label array.
    return X, y


def validate_required_classes(dataset_path):
    """
    Confirm that all six required classes can be found in the dataset folder.

    Parameters:
        dataset_path (str): Path to the dataset folder.

    Returns:
        dict: Maps the required class labels to their real folder names.

    Raises:
        ValueError: If any required class folder is missing.
    """

    # Create an empty dictionary that will map each label to a real folder name.
    class_folder_mapping = {}

    # Loop through every required class label.
    for class_label in REQUIRED_CLASSES:
        # Find the matching folder name in the dataset.
        matched_folder = find_matching_folder(dataset_path, class_label)

        # Stop the program if a required class folder cannot be found.
        if matched_folder is None:
            raise ValueError(
                f"Required class folder for '{class_label}' was not found inside '{dataset_path}'."
            )

        # Save the discovered folder mapping.
        class_folder_mapping[class_label] = matched_folder

    # Return the complete class-to-folder mapping.
    return class_folder_mapping


def run_eigenfaces_recognition(X, y):
    """
    Train and evaluate a face recognition model using the Eigenfaces approach.

    Steps:
    - split the dataset into train and test sets
    - standardize the features using only the training data
    - apply PCA to extract Eigenfaces from the training data
    - train a classifier on the Eigenfaces representation
    - evaluate on the test set
    - print the final accuracy

    Parameters:
        X (np.ndarray): Image feature matrix.
        y (np.ndarray): Label array.
    """

    # Split the full dataset into training and testing sets using the required parameters.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Create a scaler for feature standardization.
    scaler = StandardScaler()

    # Fit the scaler on the training set only, then transform the training set.
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test set using the scaler fitted only on the training set.
    X_test_scaled = scaler.transform(X_test)

    # Choose the number of principal components for Eigenfaces.
    n_components = min(80, X_train_scaled.shape[0], X_train_scaled.shape[1])

    # Create the PCA model for Eigenfaces extraction.
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)

    # Fit PCA on the training set only and convert the training images to Eigenface features.
    X_train_eigenfaces = pca.fit_transform(X_train_scaled)

    # Convert the test images to Eigenface features using the PCA fitted on training data only.
    X_test_eigenfaces = pca.transform(X_test_scaled)

    # Create the classifier that will learn from the Eigenface features.
    classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier using training data only.
    classifier.fit(X_train_eigenfaces, y_train)

    # Predict class labels for the test set.
    y_pred = classifier.predict(X_test_eigenfaces)

    # Calculate the final recognition accuracy.
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy in the format requested by the assignment.
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print the number of training and testing images for transparency.
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Print the number of Eigenfaces used by the model.
    print(f"Number of Eigenfaces: {n_components}")


def main():
    """
    Main control function for the whole assignment pipeline.
    """

    # Build the relative path to the dataset folder.
    dataset_path = DATASET_FOLDER

    # Build the relative path to the provided Haar cascade XML file.
    cascade_path = "haarcascade_frontalface_default.xml"

    # Confirm that the dataset folder exists.
    ensure_dataset_exists(dataset_path)

    # Confirm that the Haar cascade XML file exists.
    ensure_cascade_exists(cascade_path)

    # Find the source folder that contains Ariel Sharon original images.
    ariel_source_folder_name = find_matching_folder(dataset_path, "ariel sharon")

    # Stop the program if the original Ariel Sharon folder is missing.
    if ariel_source_folder_name is None:
        raise ValueError(
            "Original Ariel Sharon folder was not found. "
            "Please make sure the dataset contains a folder for Ariel Sharon original images."
        )

    # Build the full path to the Ariel Sharon source folder.
    ariel_source_folder_path = os.path.join(dataset_path, ariel_source_folder_name)

    # Build the full path to the required cropped-output folder.
    ariel_output_folder_path = os.path.join(dataset_path, ARIEL_OUTPUT_FOLDER)

    # Choose one existing Avengers folder as the filename-style reference.
    reference_folder_name = None

    # Loop through the required class labels to find a non-Ariel folder for naming reference.
    for class_label in REQUIRED_CLASSES:
        # Skip Ariel Sharon because that class is the one we are generating now.
        if class_label == "ariel_sharon":
            continue

        # Try to find a matching folder for the current class.
        matched_folder = find_matching_folder(dataset_path, class_label)

        # If a valid folder is found, store it and stop searching.
        if matched_folder is not None:
            reference_folder_name = matched_folder
            break

    # Stop the program if no reference folder is available.
    if reference_folder_name is None:
        raise ValueError("No reference folder was found for inferring the filename style.")

    # Build the full path to the reference folder.
    reference_folder_path = os.path.join(dataset_path, reference_folder_name)

    # Run face detection and cropping for Ariel Sharon.
    saved_count = detect_and_crop_faces(
        source_folder_path=ariel_source_folder_path,
        output_folder_path=ariel_output_folder_path,
        cascade_path=cascade_path,
        reference_folder_path=reference_folder_path
    )

    # Print how many Ariel Sharon cropped face images were created.
    print(f"Cropped Ariel Sharon faces saved: {saved_count}")

    # Confirm that all six required class folders now exist for recognition.
    class_folder_mapping = validate_required_classes(dataset_path)

    # Force the recognition step to use the manually created Ariel folder for the Ariel class.
    class_folder_mapping["ariel_sharon"] = ARIEL_OUTPUT_FOLDER

    # Load the full six-class dataset.
    X, y = load_face_dataset(dataset_path, class_folder_mapping)

    # Stop the program if the dataset is too small for a valid train/test split.
    if len(X) == 0:
        raise ValueError("No images were loaded. Please check the dataset structure and image files.")

    # Run Eigenfaces recognition on the six-class dataset.
    run_eigenfaces_recognition(X, y)


# Run the main function only when this file is executed directly.
if __name__ == "__main__":
    main()
