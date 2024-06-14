import json
import os
import tensorflow as tf
import tensorflow_datasets as tfds

def get_captions_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Dictionary to store filename to captions mapping
    filename_to_captions = {}

    for item in data['annotations']:
        image_id = str(item['image_id']).zfill(12)
        filename = image_id + ".jpg"
        caption = item['caption']

        if filename in filename_to_captions:
            filename_to_captions[filename].append(caption)
        else:
            filename_to_captions[filename] = [caption]

    return filename_to_captions

def load_image(image_path, image_shape):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_shape[:-1])
    return img

def download_mscoco_2017(train_captions, val_captions, num_train=None, num_val=None):
    LOCAL_IMG_PATH = "data/raw/mscoco_images/"
    os.makedirs(LOCAL_IMG_PATH, exist_ok=True)

    def save_image(filename, image):
        try:
            if isinstance(image, np.ndarray):
                image = tf.image.encode_jpeg(tf.convert_to_tensor(image)).numpy()
            if filename in train_captions:
                caption_data = train_captions[filename]
            else:
                caption_data = val_captions[filename]
            if len(caption_data) != 5:
                caption_data = (caption_data * 5)[:5]
            local_path = os.path.join(LOCAL_IMG_PATH, filename)
            tf.io.write_file(local_path, image)
            return local_path, caption_data
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None, None

    train_files = []
    train_captions_list = []
    val_files = []
    val_captions_list = []

    train_ds, _ = tfds.load('coco/2017', split='train', with_info=True)
    val_ds = tfds.load('coco/2017', split='validation')

    for idx, item in enumerate(train_ds):
        if num_train and idx >= num_train:
            break
        filename = item['image/filename'].numpy().decode("utf-8")
        image = item['image'].numpy()
        file_path, captions = save_image(filename, image)
        if file_path and captions:
            train_files.append(file_path)
            train_captions_list.append(captions)

    for idx, item in enumerate(val_ds):
        if num_val and idx >= num_val:
            break
        filename = item['image/filename'].numpy().decode("utf-8")
        image = item['image'].numpy()
        file_path, captions = save_image(filename, image)
        if file_path and captions:
            val_files.append(file_path)
            val_captions_list.append(captions)

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_captions_list))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_captions_list))

    return train_ds, val_ds
