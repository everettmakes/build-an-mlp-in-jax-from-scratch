"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    key = jax.random.PRNGKey(seed)
    return key

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    arr = jax.random.split(key, num)
    return arr

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    arr = jax.random.normal(key, shape)
    return arr

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    return jnp.argmax(inputs[:, :num_classes], axis=1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    return (labels[:, None] == jnp.arange(num_classes)[None, :]).astype(jnp.float32)

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    d = {'W': sample_normal_matrix(key, (in_dim, out_dim)) * scale, 'b': jnp.zeros(out_dim,)}
    return d

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    n = len(layer_sizes) - 1
    keys = split_prng_key(key, n)
    arr = np.array([init_linear_layer(keys[i], layer_sizes[i], layer_sizes[i+1], scale) for i in range(n)])
    return arr

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    output = x @ layer_params['W'] + layer_params['b']
    return output

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    output = jnp.maximum(x, 0)
    return output

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    return jax.nn.softmax(logits, axis=logits.ndim-1)

# Step 12 - mlp_forward
def mlp_forward(params, x):
    for layer in params[:-1]:
        x = linear_forward(x, layer)
        x = relu_activation(x)
    return linear_forward(x, params[-1])

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    return jax.nn.log_softmax(logits, axis=-1)

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    return -jnp.mean(jnp.sum(one_hot_targets * log_softmax_logits(logits), axis=-1))

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    # TODO: compute predicted classes from logits and compare to labels
    return jnp.sum(jnp.argmax(logits, axis=-1) == labels) / len(labels)

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

