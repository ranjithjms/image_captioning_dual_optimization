import tensorflow as tf
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from tensorflow.keras.layers import Layer, Embedding, Add, MultiHeadAttention, LayerNormalization, LSTM, Dropout, Dense
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Depending on the version, these imports might need to be adjusted
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

class SeqEmbedding(Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.pos_embedding = Embedding(input_dim=max_length, output_dim=depth)
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=depth, mask_zero=True)
        self.add = Add()

    def call(self, seq):
        seq = self.token_embedding(seq)
        x = tf.range(tf.shape(seq)[1])
        x = x[tf.newaxis, :]
        x = self.pos_embedding(x)
        return self.add([seq, x])

class CausalSelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.add = Add()
        self.layernorm = LayerNormalization()

    def call(self, x):
        attn = self.mha(query=x, value=x, use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)

class CrossAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.add = Add()
        self.layernorm = LayerNormalization()

    def call(self, x, y, **kwargs):
        attn, attention_scores = self.mha(query=x, value=y, return_attention_scores=True)
        self.last_attention_scores = attention_scores
        x = self.add([x, attn])
        return self.layernorm(x)

class FeedForward(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.lstm = LSTM(units, return_sequences=True)
        self.dropout = Dropout(rate=dropout_rate)
        self.layernorm = LayerNormalization()

    def call(self, x):
        x = self.lstm(x)
        x = x + self.dropout(x)
        return self.layernorm(x)

class DecoderLayer(Layer):
    def __init__(self, units, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        in_seq, out_seq = inputs
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        out_seq = self.ff(out_seq)
        return out_seq

class TokenOutput(Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()
        self.dense = Dense(units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.bias = None

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = {name: id for id, name in enumerate(self.tokenizer.get_vocabulary())}
        for tokens in tqdm.tqdm(ds):
            counts.update(tokens.numpy().flatten())
        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())
        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0
        total = counts_arr.sum()
        p = counts_arr / total
        p[counts_arr == 0] = 1.0
        log_p = np.log(p)
        entropy = -(log_p * p).sum()
        print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")
        self.bias = log_p
        self.bias[counts_arr == 0] = -1e9

    def call(self, x):
        x = self.dense(x)
        return x + self.bias

# 1. 'composite_loss'
# 2. 'reinforcement_learning'
# 3. 'gradient_approximation'
# 4. 'optimization_algorithm'

OPTIMIZATION_METHOD = 'optimization_algorithm'  # or another method based on your use case

class Captioner(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=4,
                 units=256, max_length=50, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            oov_token="[UNK]")  # Ensure there's an OOV token
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True,
            oov_token="[UNK]")  # Ensure there's an OOV token for the inverse lookup as well

        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(),
            depth=units,
            max_length=max_length)

        self.decoder_layers = [
            DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
            for n in range(num_layers)]

        self.output_layer = output_layer

    def call(self, inputs):
        image, txt = inputs
        if image.shape[-1] == 3:  # RGB images
            image = self.feature_extractor(image)
        image = einops.rearrange(image, 'b h w c -> b (h w) c')
        if isinstance(txt, tf.Tensor) and txt.dtype == tf.string:
            txt = self.tokenizer(txt)
        txt = self.seq_embedding(txt)
        for dec_layer in self.decoder_layers:
            txt = dec_layer([image, txt])
        return self.output_layer(txt)

    def train_step(self, data, optimization_method='optimization_algorithm'):
      if OPTIMIZATION_METHOD == 'composite_loss':
          return self._train_step_with_composite_loss(data)
      elif OPTIMIZATION_METHOD == 'reinforcement_learning':
          return self._train_step_with_reinforcement_learning(data)
      elif OPTIMIZATION_METHOD == 'gradient_approximation':
          return self._train_step_with_gradient_approximation(data)
      elif OPTIMIZATION_METHOD == 'optimization_algorithm':
          return self._train_step_with_optimization_algorithm(data)
      else:
          raise ValueError("Invalid optimization method")

    def calculate_bleu_score_proxy(self, true_captions, pred_captions):
      # Ensure captions are in tensor format, no explicit dtype conversion
      # Convert logits to token indices for predicted captions
      pred_captions = tf.argmax(pred_captions, axis=-1)
      true_captions = tf.convert_to_tensor(true_captions)
      pred_captions = tf.convert_to_tensor(pred_captions)
      # Clip the token indices to ensure they are within the valid range for the embedding layer
      # This step is crucial to avoid indices that are out of range
      vocab_size = self.seq_embedding.token_embedding.input_dim  # Assuming this is how you can access the vocab size
      true_captions = tf.clip_by_value(true_captions, 0, vocab_size - 1)
      pred_captions = tf.clip_by_value(pred_captions, 0, vocab_size - 1)
      if isinstance(true_captions, tf.Tensor) and true_captions.dtype == tf.string:
        true_captions = self.tokenizer(true_captions)
      if isinstance(pred_captions, tf.Tensor) and pred_captions.dtype == tf.string:
        pred_captions = self.tokenizer(pred_captions)
      # Convert token sequences to embeddings
      true_embeddings = self.seq_embedding(true_captions)  # Shape: [batch_size, seq_len, embedding_dim]
      pred_embeddings = self.seq_embedding(pred_captions)  # Shape: [batch_size, seq_len, embedding_dim]

      # Normalize embeddings to unit vectors
      true_embeddings = tf.nn.l2_normalize(true_embeddings, axis=-1)
      pred_embeddings = tf.nn.l2_normalize(pred_embeddings, axis=-1)

      # Compute cosine similarity
      # We calculate the dot product as we've normalized the embeddings to unit vectors.
      cosine_similarity_scores = tf.reduce_sum(tf.multiply(pred_embeddings, true_embeddings), axis=-1)

      # Compute a BLEU score proxy using the cosine similarity
      bleu_score_proxy = tf.reduce_mean(cosine_similarity_scores, axis=-1)  # Averaging across the sequence length dimension

      # Final BLEU proxy score for the batch
      final_bleu_score_proxy = tf.reduce_mean(bleu_score_proxy)  # Mean over the batch size

      return final_bleu_score_proxy

    # Implement the BLEU score evaluation as part of the model
    def evaluate_bleu_score(self, dataset, sample_size=100):
        total_bleu_score = 0.0
        for i, (image, true_caption) in enumerate(dataset.unbatch().take(sample_size)):
            pred_caption = self.generate_caption(image)
            true_caption_words = [self.tokenizer.index_word[token] for token in true_caption if token != 0]
            bleu_score = sentence_bleu([true_caption_words], pred_caption.split())
            total_bleu_score += bleu_score
        avg_bleu_score = total_bleu_score / sample_size
        print(f"Average BLEU Score: {avg_bleu_score}")
        return avg_bleu_score

    def _train_step_with_composite_loss(self, data):
      images, captions = data
      with tf.GradientTape() as tape:
          predictions = self(images, training=True)
          composite_loss = self.compute_composite_loss(captions, predictions)

      gradients = tape.gradient(composite_loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      self.compiled_metrics.update_state(captions, predictions)
      return {m.name: m.result() for m in self.metrics}

    def compute_composite_loss(self, true_captions, pred_captions):
      # Compute the conventional loss
      loss = self.compiled_loss(true_captions, pred_captions)

      # Compute the BLEU score proxy
      bleu_score_proxy = self.calculate_bleu_score_proxy(true_captions, pred_captions)

      # Combine the loss and BLEU score proxy
      composite_loss = loss + 0.1 * bleu_score_proxy

      return composite_loss

    def _train_step_with_reinforcement_learning(self, data):
      images, captions = data

      # Generate captions using the current model
      generated_captions = self.generate_captions(images)

      # Ideally, find or implement a version of your reward calculation that can operate in TensorFlow
      # This may involve significant changes depending on what operations are required

      # For now, let's conceptualize it as keeping the processing in TensorFlow
      # (The actual implementation will depend heavily on what you're trying to achieve)
      rewards = self.calculate_rewards(captions, generated_captions)

      with tf.GradientTape() as tape:
          logits = self(images, training=True)
          loss = self.compute_policy_gradient_loss(logits, captions, rewards)

      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return {'loss': loss}

    def custom_pad_sequences(self, sequences, padding='post', maxlen=None):
      # Convert RaggedTensor to regular Tensor if necessary
      if isinstance(sequences, tf.RaggedTensor):
          sequences = sequences.to_tensor()

      # Now, sequences is a regular Tensor, and you can use shape as usual
      if maxlen is None:
          maxlen = tf.reduce_max(tf.reduce_sum(tf.cast(sequences != 0, tf.int32), axis=1))

      sequence_length = tf.shape(sequences)[1]
      padding_amount = maxlen - sequence_length

      # Apply padding
      if padding == 'post':
          padded_sequences = tf.pad(sequences, [[0, 0], [0, padding_amount]], constant_values=0)
      else:  # 'pre'
          padded_sequences = tf.pad(sequences, [[0, 0], [padding_amount, 0]], constant_values=0)

      return padded_sequences

    def pad_ragged_to_uniform(self, tensor_ragged, padding_value=0):
      """Converts a RaggedTensor to a uniformly padded tensor."""
      return tensor_ragged.to_tensor(default_value=padding_value)

    def calculate_simple_accuracy(self, true_captions, generated_captions, padding_value=0):
      # Convert RaggedTensor to uniform tensor by padding
      generated_captions_padded = generated_captions.to_tensor(default_value=padding_value)

      # Find the maximum caption length across both true and generated captions
      max_len_true = tf.shape(true_captions)[1]
      max_len_generated = tf.shape(generated_captions_padded)[1]
      max_caption_length = tf.maximum(max_len_true, max_len_generated)

      # Pad both sets of captions to the maximum length found
      true_captions_padded = tf.pad(true_captions, [[0, 0], [0, max_caption_length - max_len_true]], constant_values=padding_value)
      generated_captions_padded = tf.pad(generated_captions_padded, [[0, 0], [0, max_caption_length - max_len_generated]], constant_values=padding_value)

      # Now, both true_captions_padded and generated_captions_padded should have the same shape
      # Perform element-wise comparison
      matches = tf.cast(tf.equal(true_captions_padded, generated_captions_padded), tf.float32)

      # Calculate simple accuracy as the mean of matches across all positions and all examples in the batch
      accuracy = tf.reduce_mean(matches)
      return accuracy

    def calculate_rewards(self, true_captions, generated_captions):
      # Tokenize generated captions if they're in string format
      # Assuming self.tokenizer can handle string tensors
      generated_captions_tokenized = self.tokenizer(generated_captions)

      # Ensure true_captions are tensors of the correct type
      true_captions = tf.cast(true_captions, dtype=tf.int64)

      # Pad both sets of captions to ensure they are the same length
      true_captions_padded = self.custom_pad_sequences(true_captions, padding='post')
      generated_captions_padded = self.custom_pad_sequences(generated_captions_tokenized, padding='post', maxlen=tf.shape(true_captions_padded)[1])
      true_captions_padded = tf.cast(true_captions_padded, dtype=tf.int32)
      generated_captions_padded = tf.cast(generated_captions_tokenized, dtype=tf.int32)

      # Now both should be in compatible formats for comparison
      rewards = self.calculate_simple_accuracy(true_captions_padded, generated_captions_padded)

      # Further processing to calculate the reward based on matches...
      return rewards

    def generate_captions(self, features, temperature=1.0):
      if isinstance(features, tuple) or isinstance(features, list):
          features = features[0]  # Adjust based on the actual structure of your data

      batch_size = tf.shape(features)[0]

      start_token_index = self.word_to_index(['[START]'])[0]  # This returns a tensor
      initial = tf.fill([batch_size], start_token_index)  # `start_token_index` determines the dtype
      initial = tf.cast(initial, tf.int32)  # Cast if necessary, but it might already be int32
      initial = tf.expand_dims(initial, axis=1)

      img_features = features  # Directly use extracted features

      tokens = initial
      for _ in range(50):  # Assuming a maximum caption length of 50
          maxval_casted = tf.cast(self.word_to_index.vocabulary_size(), tf.int32)
          next_tokens = tf.random.uniform([batch_size, 1], minval=1, maxval=maxval_casted, dtype=tf.int32)

          tokens = tf.cast(tokens, dtype=tf.int32)  # Ensure 'tokens' is int32
          # Assume 'next_tokens' is already int32 as per your token generation logic
          tokens = tf.concat([tokens, next_tokens], axis=1)

      # Convert token indices back to words
      generated_captions = tf.strings.reduce_join(self.index_to_word(tokens - 1), axis=-1, separator=' ')

      return generated_captions

    def compute_policy_gradient_loss(self, logits, captions, rewards):
      vocab_size = self.seq_embedding.token_embedding.input_dim  # Assuming this accesses the vocabulary size

      # Convert logits to log probabilities
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      # Convert captions to one-hot vectors
      captions_one_hot = tf.one_hot(captions, depth=vocab_size)

      # Select the log probabilities of the taken actions
      log_probs = tf.reduce_sum(log_probs * captions_one_hot, axis=-1)

      # Ensure rewards are properly broadcasted during multiplication
      rewards = tf.cast(rewards, tf.float32)  # Ensure rewards are float for multiplication
      rewards = tf.reshape(rewards, [-1, 1])  # Reshape for broadcasting to match log_probs' shape

      # Calculate the policy gradient loss
      loss = -tf.reduce_mean(log_probs * rewards, axis=1)  # Mean across the sequence

      # Sum over sequences for the final batch loss
      loss = tf.reduce_mean(loss)  # Final mean over the batch

      return loss

    def _train_step_with_gradient_approximation(self, data):
      images, captions = data

      with tf.GradientTape() as tape:
          # Forward pass to generate predictions
          predictions = self(images, training=True) # Adjust based on your model's call signature

          # Consider a soft BLEU score proxy calculation here based on predictions and true captions
          # This requires a custom implementation
          soft_bleu_score_proxy = self.compute_soft_bleu_score_proxy(predictions, captions)

          # Compute loss as negative of soft BLEU score proxy
          loss = -soft_bleu_score_proxy

      gradients = tape.gradient(loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return {'loss': loss}

    def compute_soft_bleu_score_proxy(self, pred_logits, true_captions):
      # Assuming pred_logits shape is [batch_size, seq_len, vocab_size]

      # Softmax to get probabilities from logits
      pred_probs = tf.nn.softmax(pred_logits, axis=-1)

      # Utilize token embeddings for predicted probabilities
      # Assuming self.seq_embedding.token_embedding is accessible and contains the embedding matrix
      token_embeddings_matrix = self.seq_embedding.token_embedding.embeddings

      # Calculate weighted sum of token embeddings based on predicted probabilities
      pred_embeddings = tf.tensordot(pred_probs, token_embeddings_matrix, axes=[[2], [0]])
      # Now, pred_embeddings should have the shape [batch_size, seq_len, embedding_dim]

      # Convert true captions to embeddings via SeqEmbedding
      # The SeqEmbedding's call method is designed to handle sequences directly
      true_embeddings = self.seq_embedding(true_captions)

      # Normalize embeddings
      true_embeddings_normalized = tf.nn.l2_normalize(true_embeddings, axis=-1)
      pred_embeddings_normalized = tf.nn.l2_normalize(pred_embeddings, axis=-1)

      # Compute cosine similarity
      cosine_similarity_scores = tf.reduce_sum(tf.multiply(pred_embeddings_normalized, true_embeddings_normalized), axis=-1)

      # Approximate BLEU score proxy as mean cosine similarity
      bleu_score_proxy = tf.reduce_mean(cosine_similarity_scores, axis=-1)

      return bleu_score_proxy

    def _train_step_with_optimization_algorithm(self, data):
      # Standard training operations or customized logic specific to "optimization_algorithm"
      # If it's similar to standard training, you can use TensorFlow's default train_step logic
      return super().train_step(data)

class OptimizationProblem(Problem):
    def __init__(self, model, data, *args, **kwargs):
        super().__init__(n_var=model.count_params(),
                         n_obj=2,  # Assuming two objectives: loss and BLEU score
                         n_constr=0,  # Number of constraints
                         xl=-5,  # Lower bounds for each variable
                         xu=5,   # Upper bounds for each variable
                         elementwise_evaluation=True)
        self.model = model
        self.data = data

    def _evaluate(self, x, out, *args, **kwargs):
        # This function should set the weights of the model to x,
        # evaluate the model on the data, and set the objectives.
        # Note: You will need to convert x into the appropriate shape for the model weights.
        self.model.set_weights(np.split(x, self.model.count_params()))
        loss, bleu_score = self.evaluate_model()  # Define this function based on your model evaluation
        out["F"] = np.array([loss, -bleu_score], dtype=np.float)  # Assuming you want to maximize BLEU score

def evaluate_model(model, data):
    # This function needs to evaluate your model's performance (loss and BLEU score)
    # on your dataset. This could involve running the model on the data and calculating
    # the metrics you're interested in.
    # Note: Ensure this function operates outside TensorFlow's automatic differentiation.
    pass

def optimize_model(model, dataset):
    problem = OptimizationProblem(model, dataset)

    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100),
                   seed=1,
                   verbose=True)

    # After optimization, you might want to update your model's weights
    # to the best solution found
    best_solution = res.X[np.argmin(res.F[:, 0])]
    model.set_weights(np.split(best_solution, model.count_params()))

class CustomOptimizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, dataset, optimization_interval=1):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.optimization_interval = optimization_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.optimization_interval == 0:
            print("Running custom optimization at the end of epoch", epoch + 1)
            optimize_model(self.model, self.dataset)
            print("Optimization completed.")

