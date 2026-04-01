"""
Face Detection and Face Recognition Project

This script finishes two tasks for the assignment.

The first part detects and crops Ariel Sharon's face images from the raw folder
using the provided Haar cascade file.

The second part performs face recognition on six classes with the Eigenfaces
approach and prints the final accuracy.

The script is meant to stay at the same directory level as the Faces folder,
so all paths are kept relative.
"""

import os
import re
import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# All images are resized to the same shape before recognition.
IMAGE_SIZE = (100, 100)

DATASET_FOLDER = "Faces"
ARIEL_OUTPUT_FOLDER = "ariel_sharon by hands"

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
    Normalize folder names so small naming differences do not break the script.
    """
    text = text.lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dataset_exists(dataset_path):
    """
    Make sure the dataset folder is available before the program starts.
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_path}' was not found. "
            f"Please place this script in the same level as the '{DATASET_FOLDER}' folder."
        )


def ensure_cascade_exists(cascade_path):
    """
    Check that the provided Haar cascade file is in the current directory.
    """
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(
            f"Cascade file '{cascade_path}' was not found. "
            f"Please keep 'haarcascade_frontalface_default.xml' in the same folder as this script."
        )


def list_subfolders(folder_path):
    """
    Return all subfolders inside a given directory.
    """
    subfolders = []

    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        if os.path.isdir(item_path):
            subfolders.append(item_name)

    return subfolders


def find_matching_folder(dataset_path, target_name):
    """
    Find a folder whose normalized name matches the requested label.
    """
    # Normalize names first so small folder naming differences do not break matching.
    normalized_target = normalize_text(target_name)
    all_subfolders = list_subfolders(dataset_path)

    for folder_name in all_subfolders:
        if normalize_text(folder_name) == normalized_target:
            return folder_name

    return None


def find_ariel_source_folder(dataset_path):
    """
    Find the folder that stores Ariel Sharon's original images.

    The exact name may vary a little across datasets, so a few common forms
    are checked here.
    """
    candidate_names = [
        "ariel sharon raw",
        "ariel_sharon_raw",
        "ariel sharon",
        "ariel_sharon"
    ]

    normalized_candidates = [normalize_text(name) for name in candidate_names]
    all_subfolders = list_subfolders(dataset_path)

    for folder_name in all_subfolders:
        if normalize_text(folder_name) in normalized_candidates:
            return folder_name

    return None


def get_image_files(folder_path):
    """
    Collect supported image files from a folder and return them in sorted order.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):
            image_files.append(file_name)

    image_files.sort()
    return image_files


def infer_filename_pattern(reference_folder_path, default_prefix="image"):
    """
    Look at one existing class folder and reuse its filename style as much as possible.

    This is mainly used so the cropped Ariel Sharon images do not look completely
    different from the other class folders.
    """
    reference_files = get_image_files(reference_folder_path)

    extension = ".jpg"
    number_width = 3
    prefix = default_prefix

    if len(reference_files) > 0:
        sample_name = reference_files[0]
        name_part, extension = os.path.splitext(sample_name)

        match = re.match(r"^(.*?)(\d+)$", name_part)
        if match:
            prefix = match.group(1)
            number_width = len(match.group(2))
        else:
            prefix = name_part + "_"

    return extension, number_width, prefix


def build_filename(prefix, index, number_width, extension):
    """
    Build one output filename using the chosen prefix, number width, and extension.
    """
    number_text = str(index).zfill(number_width)
    return f"{prefix}{number_text}{extension}"


def detect_and_crop_faces(source_folder_path, output_folder_path, cascade_path, reference_folder_path):
    """
    Detect faces from Ariel Sharon's raw images, crop the main face, resize it,
    and save it into the required output folder.

    If multiple faces are detected in one image, the largest region is used.
    """
    os.makedirs(output_folder_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cascade_path)
    image_files = get_image_files(source_folder_path)

    extension, number_width, _ = infer_filename_pattern(
        reference_folder_path,
        default_prefix="image_"
    )

    save_prefix = "ariel_sharon_"
    saved_count = 0
    output_index = 1

    for image_file in image_files:
        image_path = os.path.join(source_folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda rectangle: rectangle[2] * rectangle[3])
        face_crop = gray_image[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, IMAGE_SIZE)

        output_file_name = build_filename(save_prefix, output_index, number_width, extension)
        output_file_path = os.path.join(output_folder_path, output_file_name)

        cv2.imwrite(output_file_path, face_crop)

        saved_count += 1
        output_index += 1

    return saved_count


def validate_required_classes(dataset_path):
    """
    Build the mapping from class labels to real folder names.

    Ariel Sharon is handled a little differently here because recognition should
    use the cropped folder instead of the raw image folder.
    """
    class_folder_mapping = {}

    ariel_cropped_folder_path = os.path.join(dataset_path, ARIEL_OUTPUT_FOLDER)
    if not os.path.isdir(ariel_cropped_folder_path):
        raise ValueError(
            f"Required cropped folder '{ARIEL_OUTPUT_FOLDER}' was not found inside '{dataset_path}'."
        )

    # Use the cropped Ariel Sharon folder for recognition instead of the raw images.
    class_folder_mapping["ariel_sharon"] = ARIEL_OUTPUT_FOLDER

    for class_label in REQUIRED_CLASSES:
        if class_label == "ariel_sharon":
            continue

        matched_folder = find_matching_folder(dataset_path, class_label)
        if matched_folder is None:
            raise ValueError(
                f"Required class folder for '{class_label}' was not found inside '{dataset_path}'."
            )

        class_folder_mapping[class_label] = matched_folder

    return class_folder_mapping


def load_face_dataset(dataset_path, class_folder_mapping):
    """
    Load all six classes into feature and label arrays.

    Each image is read in grayscale, resized to a common shape, and flattened
    into a one-dimensional vector.
    """
    X = []
    y = []

    for class_label in REQUIRED_CLASSES:
        folder_name = class_folder_mapping[class_label]
        folder_path = os.path.join(dataset_path, folder_name)
        image_files = get_image_files(folder_path)

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            image_vector = image.flatten()

            X.append(image_vector)
            y.append(class_label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def run_eigenfaces_recognition(X, y):
    """
    Run the recognition stage with PCA-based Eigenfaces.

    The training set is the only part used to fit the scaler, PCA model,
    and classifier. The test set is kept separate for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    # Fit the scaler only on training data to avoid leaking test-set information
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = min(80, X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)

    # PCA is used here to generate the Eigenfaces representation
    X_train_eigenfaces = pca.fit_transform(X_train_scaled)
    X_test_eigenfaces = pca.transform(X_test_scaled)

    classifier = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    classifier.fit(X_train_eigenfaces, y_train)

    y_pred = classifier.predict(X_test_eigenfaces)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of Eigenfaces: {n_components}")


def main():
    """
    Run the full pipeline from face cropping to final recognition.
    """
    dataset_path = DATASET_FOLDER
    cascade_path = "haarcascade_frontalface_default.xml"

    ensure_dataset_exists(dataset_path)
    ensure_cascade_exists(cascade_path)

    ariel_source_folder_name = find_ariel_source_folder(dataset_path)
    if ariel_source_folder_name is None:
        raise ValueError(
            "Original Ariel Sharon folder was not found. "
            "Please make sure the dataset contains Ariel Sharon original images."
        )

    ariel_source_folder_path = os.path.join(dataset_path, ariel_source_folder_name)
    ariel_output_folder_path = os.path.join(dataset_path, ARIEL_OUTPUT_FOLDER)

    reference_folder_name = None
    for class_label in REQUIRED_CLASSES:
        if class_label == "ariel_sharon":
            continue

        matched_folder = find_matching_folder(dataset_path, class_label)
        if matched_folder is not None:
            reference_folder_name = matched_folder
            break

    if reference_folder_name is None:
        raise ValueError("No reference folder was found for filename pattern inference.")

    reference_folder_path = os.path.join(dataset_path, reference_folder_name)

    saved_count = detect_and_crop_faces(
        source_folder_path=ariel_source_folder_path,
        output_folder_path=ariel_output_folder_path,
        cascade_path=cascade_path,
        reference_folder_path=reference_folder_path
    )

    print(f"Cropped Ariel Sharon faces saved: {saved_count}")

    class_folder_mapping = validate_required_classes(dataset_path)
    X, y = load_face_dataset(dataset_path, class_folder_mapping)

    if len(X) == 0:
        raise ValueError("No images were loaded. Please check the dataset structure and image files.")

    run_eigenfaces_recognition(X, y)


if __name__ == "__main__":
    main()
