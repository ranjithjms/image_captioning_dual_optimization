import tensorflow as tf
from data_loader import load_image, get_captions_from_json, download_mscoco_2017
from model import Captioner, SeqEmbedding, TokenOutput, DecoderLayer, FeedForward, CrossAttention, CausalSelfAttention
import os

EPOCHS = 10
NUM_TRAIN = 25000
NUM_VAL = 2500
IMAGE_SHAPE = (224, 224, 3)
TRAIN_CACHE = 'data/processed/train_cache'
TEST_CACHE = 'data/processed/test_cache'
SAVE_DATASET = False

train_captions_path = "data/raw/annotations/captions_train2017.json"
val_captions_path = "data/raw/annotations/captions_val2017.json"

train_captions = get_captions_from_json(train_captions_path)
val_captions = get_captions_from_json(val_captions_path)

train_ds, val_ds = download_mscoco_2017(train_captions, val_captions, num_train=NUM_TRAIN, num_val=NUM_VAL)

visual_encoder = tf.keras.applications.EfficientNetV2S(
    input_shape=IMAGE_SHAPE,
    include_top=False,
    weights='imagenet',  
    pooling=None  
)
visual_encoder.trainable = False

tokenizer = tf.keras.layers.TextVectorization(max_tokens=5000, standardize=lambda s: tf.strings.join(['[START]', s, '[END]'], separator=' '), ragged=True)
captions_list = [txt[0].numpy().decode("utf-8") for fp, txt in train_ds]
tokenizer.adapt(captions_list)

output_layer = TokenOutput(tokenizer)
output_layer.adapt(train_ds.map(lambda inputs, labels: labels))

model = Captioner(tokenizer, feature_extractor=visual_encoder, output_layer=output_layer, num_layers=4, units=256, dropout_rate=0.5, num_heads=4)

def masked_loss(labels, preds):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)
    mask = (labels != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)
    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_acc(labels, preds):
    mask = tf.cast(labels != 0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
    return acc

class GenerateText(tf.keras.callbacks.Callback):
    def __init__(self):
        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
        self.image = load_image(image_path, IMAGE_SHAPE)

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        for t in (0.0, 0.5, 1.0):
            result = self.model.simple_gen(self.image, temperature=t)
            print(result)
        print()

callbacks = [
    GenerateText(),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=masked_loss, metrics=[masked_acc])

history = model.fit(
    train_ds.repeat(),
    steps_per_epoch=100,
    validation_data=val_ds.repeat(),
    validation_steps=20,
    epochs=EPOCHS,
    callbacks=callbacks
)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['masked_acc'], label='accuracy')
plt.plot(history.history['val_masked_acc'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
