from keras.models import load_model
from keras.layers import *
import os
import PIL
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
from itertools import product
from imageio import imwrite,imread
import gc
from IPython.display import Image,display

def get_dataset(batch_size=1):
    from glob import glob

    files = glob(os.path.join(input_dir, "*") )
    num_images = len(files)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(num_images)
    ds = ds.repeat()

    def fn(filename):
        x = tf.io.read_file(filename)
        x = tf.image.decode_jpeg(x, channels=3)
        x = tf.image.resize(x,(256,256))
        img = tf.cast(x, tf.float32) / 127.5 - 1
        #print("\n tipo img = ",type(img),"\n tipo x = ",type(x), filename, type(filename))
        return img

    ds = ds.map(fn, batch_size)
    ds = ds.batch(batch_size)
    
    steps = int(np.ceil(num_images/batch_size))
    # user iter(ds) to avoid generating iterator every epoch
    return iter(ds), steps

def loadmodel(problem, checkpoint=False):
    models_dir="models"
    if problem=="hayao":
        model_name="convTrans2d_batchNorm_spirited_away_generator_100ep_06-03_15-09_om=1_5_l=1_BEST_59ep"
    elif problem =="shinkai":
        model_name = "convTrans2d_batchNorm_your_name_generator_50ep_07-03_22-09_om=1_5_l=1_checkpoint_45ep"
    else:
        model_name="convTrans2d_batchNorm_paprika_generator_70ep_08-03_09-54_om=1_5_l=1_BEST_29ep"

    filename = os.path.join(models_dir, '%s.h5' %model_name)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model


def result_exist(image_path, style):
    return os.path.exists(os.path.join(output_dir, style, image_path.split("/")[-1]))


def save_transformed_image(output_image, img_filename, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transformed_image_path = os.path.join(save_dir, img_filename)

    if output_image is not None:
        image = PIL.Image.fromarray(output_image.astype("uint8"))
        image.save(transformed_image_path)

    return transformed_image_path


def show_and_save_images(batch_x, image_name, nrow=1, ncol=1,to_be_saved=True):
    if not isinstance(batch_x, np.ndarray):
        batch_x = batch_x.numpy()
    n, h, w, c = batch_x.shape
    out_arr = np.zeros([h * nrow, w * ncol, 3], dtype=np.uint8)
    for (i, j), k in zip(product(range(nrow), range(ncol)), range(n)):
        out_arr[(h * i):(h * (i+1)), (w * j):(w * (j+1))] = batch_x[k]
    path_name = os.path.join(output_dir,image_name)
    if to_be_saved:
        imwrite(path_name, out_arr)
    gc.collect()
    return path_name

STYLES = ["shinkai", "hayao", "paprika"]
VALID_EXTENSIONS = ['jpg', 'png', 'gif', 'JPG']
input_dir = "input_images"
output_dir = "output_images"
parser = argparse.ArgumentParser(description="transform real world images to specified cartoon style(s)")
parser.add_argument("--styles", nargs="+", default=[STYLES[0]],
                    help="specify (multiple) cartoon styles which will be used to transform input images.")
parser.add_argument("--all_styles", action="store_true",
                    help="set true if all styled results are desired")
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of images that will be transformed in parallel to speed up processing. "
                         "higher value like 4 is recommended if there are gpus.")
parser.add_argument("--comparison_view", type=str, default="smart",
                    choices=["smart", "horizontal", "vertical", "grid"],
                    help="specify how input images and transformed images are concatenated for easier comparison")


args = parser.parse_args()




styles = STYLES if args.all_styles else args.styles
models = list()
for style in styles:
    models.append(loadmodel(style))
image_paths = []
for ext in VALID_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))

progress_bar = tqdm(image_paths, desc='Transforming')
for image_path in progress_bar:
    image_filename = image_path.split("/")[-1]
    progress_bar.set_postfix(File=image_filename)

    related_image_paths = [image_path]
    for model, style in zip(models, styles):
        dataset, steps_per_epoch = get_dataset()
        input_image = dataset.next()
        save_dir = os.path.join(output_dir, style)
        return_existing_result = result_exist(image_path, style)
        #transformed_image_path = save_transformed_image(transformed_image, image_filename, save_dir)
        show_and_save_images(tf.cast((model(input_image)+1) * 127.5, tf.uint8), image_name=(image_filename), ncol=1,nrow=1)
        
progress_bar.close()

