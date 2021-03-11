from keras.models import load_model
from keras.layers import *
import os
import PIL
import sys
import glob
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

STYLES = ["shinkai", "hayao", "paprika"]


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


def pre_processing(image_path, style, expand_dim=True):
    input_image = PIL.Image.open(image_path).convert("RGB")

    if not args.keep_original_size:
        width, height = input_image.size
        aspect_ratio = width / height
        resized_height = min(height, args.max_resized_height)
        resized_width = int(resized_height * aspect_ratio)
        if width != resized_width:
            input_image = input_image.resize((resized_width, resized_height))

    input_image = np.asarray(input_image)
    input_image = input_image.astype(np.float32)

    input_image = input_image[:, :, [2, 1, 0]]

    if expand_dim:
        input_image = np.expand_dims(input_image, axis=0)
    return input_image

def result_exist(image_path, style):
    return os.path.exists(os.path.join(args.output_dir, style, image_path.split("/")[-1]))

def post_processing(transformed_image, style):
    if not type(transformed_image) == np.ndarray:
        transformed_image = transformed_image.numpy()
    transformed_image = transformed_image[0]
    transformed_image = transformed_image[:, :, [2, 1, 0]]
    transformed_image = transformed_image * 0.5 + 0.5
    transformed_image = transformed_image * 255
    return transformed_image

def save_transformed_image(output_image, img_filename, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transformed_image_path = os.path.join(save_dir, img_filename)

    if output_image is not None:
        image = PIL.Image.fromarray(output_image.astype("uint8"))
        image.save(transformed_image_path)

    return transformed_image_path


def save_concatenated_image(image_paths, image_folder="comparison", num_columns=2):
    images = [PIL.Image.open(i).convert('RGB') for i in image_paths]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    array = np.asarray([np.asarray(i.resize(min_shape)) for i in images])

    view = args.comparison_view
    if view == "smart":
        width, height = min_shape[0], min_shape[1]
        aspect_ratio = width / height
        grid_suitable = (len(args.styles) + 1) % num_columns == 0
        is_portrait = aspect_ratio <= 0.75
        if grid_suitable and not is_portrait:
            view = "grid"
        elif is_portrait:
            view = "horizontal"
        else:
            view = "vertical"

    if view == "horizontal":
        images_comb = np.hstack(array)
    elif view == "vertical":
        images_comb = np.vstack(array)
    elif view == "grid":
        rows = np.split(array, num_columns)
        rows = [np.hstack(row) for row in rows]
        images_comb = np.vstack([row for row in rows])

    images_comb = PIL.Image.fromarray(images_comb)
    file_name = image_paths[0].split("/")[-1]

    if args.output_dir not in image_folder:
        image_folder = os.path.join(args.output_dir, image_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_path = os.path.join(image_folder, file_name)
    images_comb.save(image_path)
    return image_path


if __name__ =="main":
    styles = STYLES if args.all_styles else args.styles
    models = list()
    for style in styles:
        models.append(loadmodel(style))
    
    
    progress_bar = tqdm(image_paths, desc='Transforming')
    for image_path in progress_bar:
        image_filename = image_path.split("/")[-1]
        progress_bar.set_postfix(File=image_filename)

        related_image_paths = [image_path]
        for model, style in zip(models, styles):
            input_image = pre_processing(image_path, style=style)
            save_dir = os.path.join(args.output_dir, style)
            return_existing_result = result_exist(image_path, style)

            if not return_existing_result:
                transformed_image = model(input_image)
                output_image = post_processing(transformed_image, style=style)
                transformed_image_path = save_transformed_image(output_image, image_filename, save_dir)
            else:
                transformed_image_path = save_transformed_image(None, image_filename, save_dir)

            related_image_paths.append(transformed_image_path)

        if not args.skip_comparison:
            save_concatenated_image(related_image_paths)
    progress_bar.close()

