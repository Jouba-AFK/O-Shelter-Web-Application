import torch
from torchvision import models, transforms
from PIL import Image
import os
import json
import time

_model = None
_preprocess = None

def _load_model_and_preprocess():
    """Lazily loads the model and preprocessing transformations, only loading them when first needed."""
    global _model, _preprocess
    if _model is None:
        print("Loading pre-trained model...")
        _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        _model.eval() 
        if torch.cuda.is_available():
            _model.to('cuda')
        print("Model loaded.")

        _preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _model, _preprocess

def _get_image_probabilities(image_path):
    """
    Gets probabilities for all classes after processing the image with the model.
    This is an internal helper function.
    """
    model, preprocess = _load_model_and_preprocess()

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities

    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")
        return None

def predict_cat_probability(image_path):
    probabilities = _get_image_probabilities(image_path)
    if probabilities is None:
        return None


    cat_indices = [281, 282, 283, 284, 285]

    cat_probability = 0.0
    for idx in cat_indices:
        if idx < len(probabilities):
            cat_probability += probabilities[idx].item()
    return cat_probability

def predict_dog_probability(image_path):
    probabilities = _get_image_probabilities(image_path)
    if probabilities is None:
        return None

    dog_indices = list(range(151, 269)) 
    
    dog_probability = 0.0
    for idx in dog_indices:
        if idx < len(probabilities):
            dog_probability += probabilities[idx].item()
    return dog_probability

JSON_FILE = "data.json"
UPLOAD_FOLDER = "static\\uploads"
CAT_THRESHOLD = 0.04
DOG_THRESHOLD = 0.8
CHECK_INTERVAL_SECONDS = 6

def load_json_data(filepath):
    """Loads JSON file data, returns an empty list if the file does not exist."""
    if not os.path.exists(filepath):
        print(f"JSON file '{filepath}' does not exist, creating an empty one.")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: Content of JSON file '{filepath}' is not a list, reinitializing.")
                return []
            return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file '{filepath}': {e}. Returning an empty list and suggesting to check the file.")
        return []

def save_json_data(filepath, data):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error: Failed to save JSON file '{filepath}': {e}")

def main_loop():
    """Main loop, monitors the JSON file and processes new images."""
    print(f"Starting to monitor for new entries in '{JSON_FILE}'...")
    print(f"Images will be loaded from the '{UPLOAD_FOLDER}' folder.")
    print(f"Cat detection probability threshold: {CAT_THRESHOLD}")

    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created '{UPLOAD_FOLDER}' folder.")

    processed_filenames = set() 
    last_json_mtime = 0 

    while True:
        try:
            current_mtime = os.path.getmtime(JSON_FILE)

            if current_mtime > last_json_mtime:
                print(f"\nUpdate detected in '{JSON_FILE}', processing...")
                last_json_mtime = current_mtime

                all_entries = load_json_data(JSON_FILE)
                updated_entries = [] 
                new_items_found = False

                for entry in all_entries:
                    filename = entry.get("filename")
                    timestamp = entry.get("timestamp")

                    if not filename:
                        print(f"Warning: Skipping an entry missing 'filename': {entry}")
                        continue

                    
                    if filename in processed_filenames:
                        updated_entries.append(entry) 
                        continue
                    
                    new_items_found = True
                    image_full_path = os.path.join(UPLOAD_FOLDER, filename)
                    print(f"New entry found: '{filename}' (Timestamp: {timestamp})")
                    print(f"Detecting image: {image_full_path}")


                    cat_prob = predict_cat_probability(image_full_path)
                    dog_prob = predict_dog_probability(image_full_path)

                    if cat_prob is not None and dog_prob is not None:
                        print(f"Probability of a cat in '{filename}': {cat_prob:.4f}")
                        print(f"Probability of a dog in '{filename}': {dog_prob:.4f}")

                        if cat_prob > CAT_THRESHOLD or dog_prob > DOG_THRESHOLD:
                            print(f"**Image '{filename}' identified as a cat or dog, retaining entry.**")
                            updated_entries.append(entry) 
                        else:
                            print(f"Image '{filename}' not identified as a cat or dog, deleting entry.")

                    else:
                        print(f"Could not process image '{filename}', retaining entry for later check.")
                        updated_entries.append(entry) 


                    processed_filenames.add(filename)
                
                
                if new_items_found:
                    save_json_data(JSON_FILE, updated_entries)
                    print(f"'{JSON_FILE}' updated.")
                else:
                    print("No new entries found for processing.")

            else:
                pass 

        except FileNotFoundError:
            print(f"Error: '{JSON_FILE}' or '{UPLOAD_FOLDER}' not found. Please ensure files and folders are set up correctly.")
            load_json_data(JSON_FILE) 
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
                print(f"Created '{UPLOAD_FOLDER}' folder.")
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS) 

if __name__ == "__main__":
    main_loop()