"""
Face Detection and Face Recognition Project

This script completes two tasks required by the project:

1. Face Detection
   - Use the provided Haar cascade XML file
   - Detect faces from the original Ariel Sharon images
   - Crop the detected faces
   - Save the cropped faces into:
     Faces/ariel sharon by hands

2. Face Recognition
   - Load six face categories
   - Split the dataset with test_size=0.3 and random_state=42
   - Train the model using only the training set
   - Use the Eigenfaces method for recognition
   - Print the final accuracy

Important:
- This file must stay in the same directory level as the Faces folder.
- Do not use absolute paths.
"""

# Import os for file and folder operations.
import os

# Import re for simple filename pattern handling.
import re

# Import OpenCV for image reading, face detection, image conversion, and saving.
import cv2

# Import NumPy for array operations.
import numpy as np

# Import PCA for the Eigenfaces step.
from sklearn.decomposition import PCA

# Import train_test_split for splitting the dataset.
from sklearn.model_selection import train_test_split

# Import StandardScaler to normalize feature values before PCA.
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

# Import accuracy_score to evaluate the recognition result.
from sklearn.metrics import accuracy_score


# Define the fixed image size used for all faces.
IMAGE_SIZE = (100, 100)

# Define the root dataset folder.
DATASET_FOLDER = "Faces"

# Define the name of the required cropped Ariel Sharon folder.
ARIEL_OUTPUT_FOLDER = "ariel sharon by hands"

# Define the six required categories for face recognition.
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
    Normalize folder names or labels so different naming styles can still match.

    Examples:
    - "Ariel Sharon" -> "ariel sharon"
    - "ariel_sharon" -> "ariel sharon"
    - "Robert-Downey-Jr" -> "robert downey jr"

    Parameters:
        text (str): Original text.

    Returns:
        str: Normalized text.
    """

    # Convert the text to lowercase.
    text = text.lower()

    # Replace underscores and hyphens with spaces.
    text = text.replace("_", " ").replace("-", " ")

    # Remove repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    # Return the cleaned text.
    return text


def ensure_dataset_exists(dataset_path):
    """
    Check whether the dataset folder exists.

    Parameters:
        dataset_path (str): Relative path to the dataset folder.

    Raises:
        FileNotFoundError: If the dataset folder does not exist.
    """

    # Stop the program if the dataset folder is missing.
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_path}' was not found. "
            f"Please place this script in the same level as the '{DATASET_FOLDER}' folder."
        )


def ensure_cascade_exists(cascade_path):
    """
    Check whether the Haar cascade XML file exists.

    Parameters:
        cascade_path (str): Relative path to the XML file.

    Raises:
        FileNotFoundError: If the XML file does not exist.
    """

    # Stop the program if the Haar cascade file is missing.
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(
            f"Cascade file '{cascade_path}' was not found. "
            f"Please keep 'haarcascade_frontalface_default.xml' in the same folder as this script."
        )


def list_subfolders(folder_path):
    """
    Return all subfolder names inside a folder.

    Parameters:
        folder_path (str): Path to the parent folder.

    Returns:
        list[str]: A list of subfolder names.
    """

    # Create an empty list to store folder names.
    subfolders = []

    # Loop through every item inside the folder.
    for item_name in os.listdir(folder_path):
        # Build the full path of the current item.
        item_path = os.path.join(folder_path, item_name)

        # Keep the item only if it is a folder.
        if os.path.isdir(item_path):
            subfolders.append(item_name)

    # Return the folder list.
    return subfolders


def find_matching_folder(dataset_path, target_name):
    """
    Find a folder whose normalized name matches the target label exactly.

    Parameters:
        dataset_path (str): Path to the dataset folder.
        target_name (str): Target label name.

    Returns:
        str or None: The matched real folder name, or None if not found.
    """

    # Normalize the target label.
    normalized_target = normalize_text(target_name)

    # Get all dataset subfolders.
    all_subfolders = list_subfolders(dataset_path)

    # Loop through all discovered subfolders.
    for folder_name in all_subfolders:
        # Normalize the current folder name.
        normalized_folder = normalize_text(folder_name)

        # Return the real folder name if it matches exactly.
        if normalized_folder == normalized_target:
            return folder_name

    # Return None if no folder matches.
    return None


def find_ariel_source_folder(dataset_path):
    """
    Find the original Ariel Sharon image folder.

    The raw folder may use names such as:
    - Ariel Sharon
    - ariel_sharon
    - Ariel Sharon Raw
    - ariel_sharon_raw

    Parameters:
        dataset_path (str): Path to the dataset folder.

    Returns:
        str or None: The matched raw Ariel Sharon folder name, or None if not found.
    """

    # Define acceptable names for the original Ariel Sharon image folder.
    candidate_names = [
        "ariel sharon raw",
        "ariel_sharon_raw",
        "ariel sharon",
        "ariel_sharon"
    ]

    # Normalize all acceptable names for comparison.
    normalized_candidates = [normalize_text(name) for name in candidate_names]

    # Get all subfolders inside the dataset folder.
    all_subfolders = list_subfolders(dataset_path)

    # Loop through each real folder in the dataset.
    for folder_name in all_subfolders:
        # Normalize the current real folder name.
        normalized_folder = normalize_text(folder_name)

        # Return the folder if it matches one of the accepted raw Ariel names.
        if normalized_folder in normalized_candidates:
            return folder_name

    # Return None if no valid Ariel source folder is found.
    return None


def get_image_files(folder_path):
    """
    Return valid image filenames from a folder.

    Parameters:
        folder_path (str): Path to an image folder.

    Returns:
        list[str]: Sorted image filenames.
    """

    # Define supported image extensions.
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    # Create an empty list to store image filenames.
    image_files = []

    # Loop through all files in the folder.
    for file_name in os.listdir(folder_path):
        # Keep the file only if it is a supported image type.
        if file_name.lower().endswith(valid_extensions):
            image_files.append(file_name)

    # Sort the filenames for stable processing order.
    image_files.sort()

    # Return the sorted list.
    return image_files


def infer_filename_pattern(reference_folder_path, default_prefix="image"):
    """
    Infer the filename style from an existing class folder.

    This function tries to guess:
    - extension
    - number width
    - prefix

    Example:
    If the folder contains 'chris_evans001.png',
    this function can infer:
    - extension = '.png'
    - number_width = 3
    - prefix = 'chris_evans'

    Parameters:
        reference_folder_path (str): Path to a reference class folder.
        default_prefix (str): Fallback prefix if the pattern is unclear.

    Returns:
        tuple[str, int, str]:
            extension, number_width, prefix
    """

    # Get all image files from the reference folder.
    reference_files = get_image_files(reference_folder_path)

    # Use default values first.
    extension = ".jpg"
    number_width = 3
    prefix = default_prefix

    # Only inspect the folder if at least one image exists.
    if len(reference_files) > 0:
        # Use the first filename as a sample.
        sample_name = reference_files[0]

        # Split the filename into base name and extension.
        name_part, extension = os.path.splitext(sample_name)

        # Try to match a name ending with digits.
        match = re.match(r"^(.*?)(\d+)$", name_part)

        # If trailing digits exist, extract prefix and number width.
        if match:
            prefix = match.group(1)
            number_width = len(match.group(2))
        else:
            # If no trailing digits exist, keep the full name and add an underscore.
            prefix = name_part + "_"

    # Return the inferred filename pattern.
    return extension, number_width, prefix


def build_filename(prefix, index, number_width, extension):
    """
    Build an output filename from the given pattern parts.

    Parameters:
        prefix (str): Text before the number.
        index (int): Current file number.
        number_width (int): Number of digits with zero padding.
        extension (str): File extension.

    Returns:
        str: Final filename.
    """

    # Convert the index to a zero-padded string.
    number_text = str(index).zfill(number_width)

    # Return the final combined filename.
    return f"{prefix}{number_text}{extension}"


def detect_and_crop_faces(source_folder_path, output_folder_path, cascade_path, reference_folder_path):
    """
    Detect faces from Ariel Sharon raw images, crop the largest face,
    resize it, and save it into the output folder.

    Parameters:
        source_folder_path (str): Path to Ariel Sharon raw images.
        output_folder_path (str): Path to save cropped Ariel Sharon faces.
        cascade_path (str): Path to the Haar cascade XML file.
        reference_folder_path (str): Path to a reference class folder for filename style.

    Returns:
        int: Number of cropped faces successfully saved.
    """

    # Make sure the output folder exists.
    os.makedirs(output_folder_path, exist_ok=True)

    # Load the Haar cascade classifier.
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Get all image files from the Ariel Sharon raw folder.
    image_files = get_image_files(source_folder_path)

    # Infer the naming pattern from the reference folder.
    extension, number_width, _ = infer_filename_pattern(reference_folder_path, default_prefix="image_")

    # Use a consistent prefix for Ariel Sharon cropped files.
    save_prefix = "ariel_sharon_"

    # Create a counter for saved images.
    saved_count = 0

    # Create a counter for output filenames.
    output_index = 1

    # Loop through each Ariel Sharon raw image.
    for image_file in image_files:
        # Build the full path for the current image.
        image_path = os.path.join(source_folder_path, image_file)

        # Read the current image.
        image = cv2.imread(image_path)

        # Skip the file if the image cannot be read.
        if image is None:
            continue

        # Convert the image to grayscale for Haar cascade detection.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image.
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Skip the image if no face is detected.
        if len(faces) == 0:
            continue

        # Select the largest detected face.
        x, y, w, h = max(faces, key=lambda rectangle: rectangle[2] * rectangle[3])

        # Crop the face region.
        face_crop = gray_image[y:y + h, x:x + w]

        # Resize the cropped face to the fixed size.
        face_crop = cv2.resize(face_crop, IMAGE_SIZE)

        # Build the output filename.
        output_file_name = build_filename(save_prefix, output_index, number_width, extension)

        # Build the full output path.
        output_file_path = os.path.join(output_folder_path, output_file_name)

        # Save the cropped face image.
        cv2.imwrite(output_file_path, face_crop)

        # Increase the counters.
        saved_count += 1
        output_index += 1

    # Return the number of saved cropped faces.
    return saved_count


def validate_required_classes(dataset_path):
    """
    Validate that all required class folders exist for the recognition step.

    For Ariel Sharon, the recognition step must use the cropped folder
    'ariel sharon by hands', not the raw image folder.

    Parameters:
        dataset_path (str): Path to the dataset folder.

    Returns:
        dict: A mapping from required class labels to real folder names.

    Raises:
        ValueError: If a required folder cannot be found.
    """

    # Create an empty dictionary for class-to-folder mapping.
    class_folder_mapping = {}

    # Build the expected Ariel cropped folder path.
    ariel_cropped_folder_path = os.path.join(dataset_path, ARIEL_OUTPUT_FOLDER)

    # Stop the program if the cropped Ariel folder does not exist.
    if not os.path.isdir(ariel_cropped_folder_path):
        raise ValueError(
            f"Required cropped folder '{ARIEL_OUTPUT_FOLDER}' was not found inside '{dataset_path}'."
        )

    # Directly map the Ariel Sharon class to the cropped-output folder.
    class_folder_mapping["ariel_sharon"] = ARIEL_OUTPUT_FOLDER

    # Loop through every required class label.
    for class_label in REQUIRED_CLASSES:
        # Skip Ariel Sharon because it is already mapped.
        if class_label == "ariel_sharon":
            continue

        # Find the matching real folder for the current class.
        matched_folder = find_matching_folder(dataset_path, class_label)

        # Stop the program if the class folder is missing.
        if matched_folder is None:
            raise ValueError(
                f"Required class folder for '{class_label}' was not found inside '{dataset_path}'."
            )

        # Save the mapping.
        class_folder_mapping[class_label] = matched_folder

    # Return the final mapping.
    return class_folder_mapping


def load_face_dataset(dataset_path, class_folder_mapping):
    """
    Load the face dataset for all six required classes.

    Each image is:
    - read in grayscale
    - resized to IMAGE_SIZE
    - flattened into a one-dimensional vector

    Parameters:
        dataset_path (str): Path to the dataset folder.
        class_folder_mapping (dict): Mapping from class labels to folder names.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X = feature matrix
            y = label array
    """

    # Create a list to store image vectors.
    X = []

    # Create a list to store labels.
    y = []

    # Loop through each required class label.
    for class_label in REQUIRED_CLASSES:
        # Get the actual folder name for the current class.
        folder_name = class_folder_mapping[class_label]

        # Build the full folder path.
        folder_path = os.path.join(dataset_path, folder_name)

        # Get all image files in the current folder.
        image_files = get_image_files(folder_path)

        # Loop through every image file in the class folder.
        for image_file in image_files:
            # Build the full path to the image.
            image_path = os.path.join(folder_path, image_file)

            # Read the image in grayscale.
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Skip unreadable images.
            if image is None:
                continue

            # Resize the image to the fixed size.
            image = cv2.resize(image, IMAGE_SIZE)

            # Flatten the 2D image into a 1D vector.
            image_vector = image.flatten()

            # Add the image vector and label to the dataset.
            X.append(image_vector)
            y.append(class_label)

    # Convert lists to NumPy arrays.
    X = np.array(X)
    y = np.array(y)

    # Return the dataset arrays.
    return X, y


def run_eigenfaces_recognition(X, y):
    """
    Train and evaluate the face recognition model using the Eigenfaces method.

    Steps:
    - split the dataset into training and testing sets
    - standardize the features using only the training data
    - apply PCA on the training data
    - project both train and test sets into the Eigenfaces space
    - train a classifier using training data only
    - evaluate the classifier on the test set
    - print the final accuracy

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label array.
    """

    # Split the dataset into training and testing sets using the required parameters.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Create a scaler object.
    scaler = StandardScaler()

    # Fit the scaler on the training set only, then transform the training set.
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test set using the training scaler.
    X_test_scaled = scaler.transform(X_test)

    # Choose the number of principal components for Eigenfaces.
    n_components = min(80, X_train_scaled.shape[0], X_train_scaled.shape[1])

    # Create the PCA model for Eigenfaces.
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)

    # Fit PCA on the training data only and transform the training set.
    X_train_eigenfaces = pca.fit_transform(X_train_scaled)

    # Transform the test set using the PCA model trained on the training set.
    X_test_eigenfaces = pca.transform(X_test_scaled)

    # Create the classifier.
    classifier = SVC(kernel="rbf", class_weight="balanced", random_state=42)

    # Train the classifier using training data only.
    classifier.fit(X_train_eigenfaces, y_train)

    # Predict labels for the test set.
    y_pred = classifier.predict(X_test_eigenfaces)

    # Calculate the final accuracy.
    accuracy = accuracy_score(y_test, y_pred)

    # Print the required final accuracy.
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print extra information for transparency.
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of Eigenfaces: {n_components}")


def main():
    """
    Run the complete project pipeline.
    """

    # Define the relative dataset path.
    dataset_path = DATASET_FOLDER

    # Define the relative Haar cascade path.
    cascade_path = "haarcascade_frontalface_default.xml"

    # Check whether the dataset folder exists.
    ensure_dataset_exists(dataset_path)

    # Check whether the Haar cascade file exists.
    ensure_cascade_exists(cascade_path)

    # Find the Ariel Sharon raw folder.
    ariel_source_folder_name = find_ariel_source_folder(dataset_path)

    # Stop the program if the original Ariel Sharon folder is missing.
    if ariel_source_folder_name is None:
        raise ValueError(
            "Original Ariel Sharon folder was not found. "
            "Please make sure the dataset contains Ariel Sharon original images."
        )

    # Build the full path to the Ariel Sharon raw folder.
    ariel_source_folder_path = os.path.join(dataset_path, ariel_source_folder_name)

    # Build the full path to the cropped Ariel Sharon output folder.
    ariel_output_folder_path = os.path.join(dataset_path, ARIEL_OUTPUT_FOLDER)

    # Create a variable to store the folder used as filename-style reference.
    reference_folder_name = None

    # Loop through the required class labels to find a non-Ariel reference folder.
    for class_label in REQUIRED_CLASSES:
        # Skip Ariel Sharon because that class is generated from cropping.
        if class_label == "ariel_sharon":
            continue

        # Try to find the matching folder for the current class.
        matched_folder = find_matching_folder(dataset_path, class_label)

        # If a valid folder is found, keep it and stop searching.
        if matched_folder is not None:
            reference_folder_name = matched_folder
            break

    # Stop the program if no valid reference folder exists.
    if reference_folder_name is None:
        raise ValueError("No reference folder was found for filename pattern inference.")

    # Build the full path to the reference folder.
    reference_folder_path = os.path.join(dataset_path, reference_folder_name)

    # Run face detection and cropping for Ariel Sharon.
    saved_count = detect_and_crop_faces(
        source_folder_path=ariel_source_folder_path,
        output_folder_path=ariel_output_folder_path,
        cascade_path=cascade_path,
        reference_folder_path=reference_folder_path
    )

    # Print how many Ariel Sharon cropped face images were saved.
    print(f"Cropped Ariel Sharon faces saved: {saved_count}")

    # Validate all required class folders for recognition.
    class_folder_mapping = validate_required_classes(dataset_path)

    # Load the complete six-class dataset.
    X, y = load_face_dataset(dataset_path, class_folder_mapping)

    # Stop the program if no images were loaded.
    if len(X) == 0:
        raise ValueError("No images were loaded. Please check the dataset structure and image files.")

    # Run the Eigenfaces recognition step.
    run_eigenfaces_recognition(X, y)


# Run the main function only if this script is executed directly.
if __name__ == "__main__":
    main()
