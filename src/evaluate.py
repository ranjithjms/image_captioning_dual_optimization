import random
import pandas as pd
import nltk
import tensorflow as tf
from data_loader import load_image, get_captions_from_json, download_mscoco_2017
from model import Captioner, TokenOutput
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Below are hyperparameters for the evaluation
EPOCHS = 50
NUM_TRAIN = 25000
NUM_VAL = 2500
IMAGE_SHAPE = (224, 224, 3)
# Trained model weights to be stored in this directory
TRAIN_CACHE = 'data/processed/train_cache'
# Test model weights to be stored in this directory
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

def get_random_image_data(test_raw_dataset):
    random_index = random.randint(0, len(test_raw_dataset) - 1)
    random_element = list(test_raw_dataset.skip(random_index).take(1))[0]
    image_path = random_element[0].numpy().decode("utf-8")
    captions = [caption.numpy().decode("utf-8") for caption in random_element[1]]
    return image_path, captions

def calculate_bleu_scores(reference, candidate, n=4):
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for i in range(1, n + 1):
        weights = [1.0 / i] * i + [0.0] * (n - i)
        score = sentence_bleu([reference], candidate, weights=weights, smoothing_function=smoothie)
        bleu_scores.append(score)
    return bleu_scores

def evaluate_metrics(predicted_caption, original_captions, weight_factors):

    best_scores = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
    candidate_tokenized = nltk.word_tokenize(predicted_caption.lower())
    for caption in original_captions:
        reference_tokenized = nltk.word_tokenize(caption.lower())
        bleu_scores = calculate_bleu_scores(reference_tokenized, candidate_tokenized, n=4)
        adjusted_bleu_scores = [max(0, bleu_scores[i] - weight_factors[i]) for i in range(4)]
        for i, metric in enumerate(['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']):
            score = adjusted_bleu_scores[i]
            if score > best_scores[metric]:
                best_scores[metric] = score
    df = pd.DataFrame({'Evaluation Technique': list(best_scores.keys()), 'Score': [score for score in best_scores.values()]})
    return df

# Hyperparameters for the evaluation
# The weights for the BLEU scores
weight_factors = [0.0, 0.0, 0.0, 0.0]

image_path, original_captions = get_random_image_data(val_ds)
image = load_image(image_path, IMAGE_SHAPE)
predicted_caption = model.simple_gen(image, temperature=0.0)
result_df = evaluate_metrics(predicted_caption, original_captions, weight_factors)
print('Original Caption:', original_captions[0])
print('Predicted Caption: ', predicted_caption)
print(result_df)
