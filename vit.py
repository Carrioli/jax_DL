import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util
import optax
from jax import grad, jit, lax, nn, random, value_and_grad, vmap
from jax.nn import gelu, relu, relu6, selu, standardize
from optax import lion
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from load_data import LoadDataset



# constants
batch_size  = 1024
train_ratio = 0.8
img_dim = 96
patch_size = 16
n_layers = 3 # number of transformer layers
n_heads = 3 # number of transformer heads
num_tokens = (img_dim // patch_size) ** 2 + 1
D = (patch_size ** 2) * 3
d_k = D // n_heads
d_v = D // n_heads
lr = 2e-5



def init_params(initializer):
    master_key = random.PRNGKey(0)
    num_keys = 50
    keys = random.split(master_key, num=num_keys)
    params = {
        'dense1': {
            'w': initializer(keys[0], (D, D)),
            'b': jax.lax.squeeze(initializer(keys[1], (D, 1)), dimensions=(1,)) # initialized as 2D because of initializer, then squeeze
            },
        'cls_token': jax.lax.squeeze(initializer(keys[2], (D, 1)), dimensions=(1,)),
        'position_embeddings': initializer(keys[3], (num_tokens, D)),
        'transformer_block': {
            'W_Q':  initializer(keys[4], (n_layers, n_heads, D, d_k)),
            'W_K':  initializer(keys[5], (n_layers, n_heads, D, d_k)),
            'W_V':  initializer(keys[6], (n_layers, n_heads, D, d_v)),
            'W_O':      initializer(keys[7], (n_layers, n_heads * d_v, D)), # final linear layer
            'mlp_W1':   initializer(keys[8], (n_layers, D, 4 * D)),
            'mlp_W2':   initializer(keys[9], (n_layers, 4 * D, D)),
        },
        'mlp_head': {
            'dense1': {
                'w': initializer(keys[10], (D, D)),
                'b': jax.lax.squeeze(initializer(keys[11], (D, 1)), dimensions=(1,))
            },
            'dense2': {
                'w': initializer(keys[12], (D, 2)),
                'b': jax.lax.squeeze(initializer(keys[13], (2, 1)), dimensions=(1,))
            }
        }
    }
    return params


def extract_patches(x):
    C, H, W = x.shape
    x = x.reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = jax.lax.transpose(x, permutation=(1, 3, 0, 2, 4))
    patches = x.reshape(-1, C * patch_size * patch_size)
    return patches


def singlehead_self_attention(x, W_Q, W_K, W_V):
    Q = jax.lax.dot(x, W_Q)
    K = jax.lax.dot(x, W_K)
    V = jax.lax.dot(x, W_V)
    d_k = W_Q.shape[-1]
    scores = jax.lax.dot(Q, jax.lax.transpose(K, permutation=(1, 0))) / jnp.sqrt(d_k)
    attention_weights = jax.nn.softmax(scores, axis=-1)
    return jax.lax.dot(attention_weights, V)


def multihead_self_attention(x, W_Q, W_K, W_V, W_O):
    # get all the heades, concatenate them, and then apply the final linear layer
    x = jax.vmap(singlehead_self_attention, in_axes=[None, 0, 0, 0])(x, W_Q, W_K, W_V)
    # concatenate the heads so that x has shape (x.shape[0]*x.shape[1], x.shape[2])
    x = x.reshape(x.shape[1], -1)
    return jax.lax.dot(x, W_O)


def mlp(x, W1, W2):
    return (jax.lax.dot(gelu(jax.lax.dot(x, W1)), W2))


def transformer_block(params, z_l_minus_1, layer_idx):
    W_Q    = params['transformer_block']['W_Q'][layer_idx]
    W_K    = params['transformer_block']['W_K'][layer_idx]
    W_V    = params['transformer_block']['W_V'][layer_idx]
    W_O    = params['transformer_block']['W_O'][layer_idx]
    mlp_W1 = params['transformer_block']['mlp_W1'][layer_idx]
    mlp_W2 = params['transformer_block']['mlp_W2'][layer_idx]

    z_prime_l = multihead_self_attention(standardize(z_l_minus_1, axis=1), W_Q, W_K, W_V, W_O) + z_l_minus_1
    z_l = mlp(standardize(z_prime_l, axis=1), mlp_W1, mlp_W2) + z_prime_l
    return z_l, None


def model(params, x):
    x = extract_patches(x)
    x = jax.lax.dot(x, params['dense1']['w']) + params['dense1']['b']
    # Tokenization: Add the classification token
    x = jnp.concatenate((params['cls_token'][jnp.newaxis, :], x), axis=0)
    # Add learable positional encoding
    x += params['position_embeddings']

    final_carry, _ = jax.lax.scan(lambda carry, indx: transformer_block(params, carry, indx), x, jnp.arange(n_layers))

    cls_token = final_carry[0]

    x = jax.lax.dot(cls_token, params['mlp_head']['dense1']['w']) + params['mlp_head']['dense1']['b']
    x = jax.lax.dot(x, params['mlp_head']['dense2']['w']) + params['mlp_head']['dense2']['b']
    return x


def batched_model(params, x):
    return jax.vmap(model, in_axes=(None, 0))(params, x)


def batched_softmax_cross_entropy(params, x, y_true):
    y_pred = batched_model(params, x)
    y_true_one_hot = nn.one_hot(y_true, num_classes = 2, dtype = int)
    return optax.softmax_cross_entropy(logits = y_pred, labels = y_true_one_hot).mean() # loss is already vectorized 


@jax.jit
def update(params, opt_state, x, y_true):
    loss, grads = value_and_grad(batched_softmax_cross_entropy, argnums = 0)(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


@jax.jit
def eval(params, x_batch, y_batch):
    y_pred_batch = batched_model(params, x_batch)
    accuracy = jnp.mean(jnp.argmax(y_pred_batch, -1) == y_batch)
    return accuracy


def eval_fn(params, data_loader):
    accuracies = [eval(params, jnp.array(x_batch), jnp.array(y_batch)) for (x_batch, y_batch) in iter(data_loader)]
    return sum(accuracies)/len(accuracies)


def train_epoch(params, opt_state, data_loader, bar):
    epoch_loss = 0.0
    for (batch_img, batch_label) in iter(data_loader):
        x = jnp.array(batch_img) # batches need to be converted to jnp array
        y_true = jnp.array(batch_label)
        batch_loss, params, opt_state = update(params, opt_state, x, y_true)
        epoch_loss += batch_loss
        bar.update(1)
    return params, opt_state, epoch_loss


def train_and_eval(params, opt_state, train_dl, test_dl, n_epochs):
    bar = tqdm(total = n_epochs * len(train_dl), ncols = 150, leave = True)
    bar.colour = '#0000ff'
    for epoch in range(n_epochs):
        bar.set_postfix(epoch = f'{epoch + 1} out of {n_epochs}')
        params, opt_state, epoch_loss = train_epoch(params, opt_state, train_dl, bar)
        # eval for each epoch
        accuracy = eval_fn(params, test_dl)
        tqdm.write(f'Epoch: {epoch + 1}, average epoch loss: {epoch_loss/len(train_dl):.6f}. Test accuracy: {accuracy}')
    return params


if __name__ == '__main__':

    # get data
    sub_path = 'datasets/histopathologic-cancer-detection/'
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=img_dim, sample_fraction=.01, augment=True)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, prefetch_factor=1)
    test_dl  = DataLoader(test_ds , batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, prefetch_factor=1)


    # init model
    initializer = nn.initializers.lecun_normal()
    params = init_params(initializer)
    optimizer = optax.lion(learning_rate = lr)
    opt_state = optimizer.init(params)

    # print constants
    num_params = format(sum(jax_array.size for jax_array in jax.tree_util.tree_flatten(params)[0]), ',')
    print(f'Number of learnable parameters: {num_params}')
    print(f'Batch size: {batch_size}')
    print(f'Training on {len(train_ds)} examples and {len(train_dl)} batches of size {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Patch size: {patch_size}')
    print(f'Number of transformer layers: {n_layers}')
    print(f'Number of transformer heads: {n_heads}')

    # train
    trained_params = train_and_eval(params, opt_state, train_dl, test_dl, n_epochs=30)