import os
import tarfile
import urllib.request

# URL del modelo preentrenado
MODEL_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
MODEL_DIR = "ssd_mobilenet_v2"

def download_and_extract_model(model_url, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tar_file_path = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(os.path.join(model_dir, "saved_model")):
        urllib.request.urlretrieve(model_url, tar_file_path)
        with tarfile.open(tar_file_path) as tar:
            tar.extractall(path=model_dir)
        os.remove(tar_file_path)
        print("Modelo descargado y extraÃ­do en", model_dir)
    else:
        print("Modelo ya existe en", model_dir)

download_and_extract_model(MODEL_URL, MODEL_DIR)

