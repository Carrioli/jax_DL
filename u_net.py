from sys import exit

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import lax, nn, random, value_and_grad
from jax.lax import conv, conv_transpose
from jax.nn import relu, relu6, selu
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from load_data import LoadDataset


# constants
batch_size  = 16
num_epochs  = 30
train_ratio = 0.8
lr = 2e-5
img_dim = 63
model_depth = 3

class PostIncrement:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        temp = self.value
        self.value += 1
        return temp


def init_params(initializer):
    master_key = random.PRNGKey(0)
    num_keys = 100
    keys = random.split(master_key, num=num_keys)

    # width and height at the bottom of the U
    bottom_dim = (img_dim - 2 ** (model_depth + 2) + 4) // 2 ** (model_depth - 1)

    k = PostIncrement()

    # Every contracting layer has 2 conv layers
    params = {}
    params['contracting'] = [
        {
            'conv1': {
                'w': initializer(keys[k.increment()], (2 ** (i + 6), 3 if i == 0 else 2 ** (i + 5), 3, 3)),  # out_c, in_c, h, w
                'b': initializer(keys[k.increment()], (1, 2 ** (i + 6), 1, 1))
            },
            'conv2': {
                'w': initializer(keys[k.increment()], (2 ** (i + 6), 2 ** (i + 6), 3, 3)),
                'b': initializer(keys[k.increment()], (1, 2 ** (i + 6), 1, 1))
            }
        }
        for i in range(model_depth - 1)
    ]

    params['bottom'] = {
        'conv1': {
            'w': initializer(keys[k.increment()], (2 ** (model_depth + 5), 2 ** (model_depth + 4), 3, 3)),
            'b': initializer(keys[k.increment()], (1, 2 ** (model_depth + 5), 1, 1))
        },
        'conv2': {
            'w': initializer(keys[k.increment()], (2 ** (model_depth + 5), 2 ** (model_depth + 5), 3, 3)),
            'b': initializer(keys[k.increment()], (1, 2 ** (model_depth + 5), 1, 1))
        }
    }

    # Every expanding layer has 3 conv layers
    params['expanding'] = [
        {
            'conv1': {
                'w': initializer(keys[k.increment()], (2 ** (model_depth + 4 - i), 2 ** (model_depth + 5 - i), 2, 2)),
                'b': initializer(keys[k.increment()], (1, 2 ** (model_depth + 4 - i), 1, 1))
            },
            'conv2': {
                'w': initializer(keys[k.increment()], (2 ** (model_depth + 4 - i), 2 ** (model_depth + 5 - i), 3, 3)),
                'b': initializer(keys[k.increment()], (1, 2 ** (model_depth + 4 - i), 1, 1))
            },
            'conv3': {
                'w': initializer(keys[k.increment()], (2 ** (model_depth + 4 - i), 2 ** (model_depth + 4 - i), 3, 3)),
                'b': initializer(keys[k.increment()], (1, 2 ** (model_depth + 4 - i), 1, 1))
            }
        }
        for i in range(model_depth - 1)
    ]

    i = model_depth - 2
    params['final_layers'] = {
        'conv1': {
            'w': initializer(keys[k.increment()], (2, 2 ** (model_depth + 4 - i), 1, 1)),
            'b': initializer(keys[k.increment()], (1, 2, 1, 1))
        },
        'dense1': {
            'w': initializer(keys[k.increment()], (2 * (2 ** i * (2 * bottom_dim - 8) + 4)**2, 512)),
            'b': initializer(keys[k.increment()], (512, 1)).squeeze()
        },
        'dense2': {
            'w': initializer(keys[k.increment()], (512, 2)),
            'b': initializer(keys[k.increment()], (2, 1)).squeeze()
        }
    }
    return params


def contracting_layer(layer_params, x):
    w1 = layer_params['conv1']['w']
    b1 = layer_params['conv1']['b']
    w2 = layer_params['conv2']['w']
    b2 = layer_params['conv2']['b']

    x = activation(conv(x, w1, (1, 1), 'VALID') + b1)
    x = activation(conv(x, w2, (1, 1), 'VALID') + b2)

    y = jnp.copy(x)

    x = hk.max_pool(value = x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
    return x, y


def expanding_layer(layer_params, x, y):
    w1 = layer_params['conv1']['w']
    b1 = layer_params['conv1']['b']
    w2 = layer_params['conv2']['w']
    b2 = layer_params['conv2']['b']
    w3 = layer_params['conv3']['w']
    b3 = layer_params['conv3']['b']

    # Expansion
    x = conv_transpose(x, w1, (2, 2), 'VALID', dimension_numbers=('NCHW', 'OIHW', 'NCHW')) + b1

    # Concatenate
    start = (y.shape[-1] - x.shape[-1]) // 2
    end   = start + x.shape[-1]
    x = jnp.concatenate((x, y[:,:,start:end,start:end]), axis = 1)

    x = activation(conv(x, w2, (1, 1), 'VALID') + b2)
    x = activation(conv(x, w3, (1, 1), 'VALID') + b3)

    return x


def batched_model(params, x):
    #------------------------------ contracting ------------------------------
    ys = [None] * (model_depth - 1)
    for i, layer_params in enumerate(params['contracting']):
        x, ys[model_depth - 2 - i] = contracting_layer(layer_params, x) # ys are saved in reversed order

    # bottom layers
    x = activation(conv(x, params['bottom']['conv1']['w'], (1, 1), 'VALID') + params['bottom']['conv1']['b'])
    x = activation(conv(x, params['bottom']['conv2']['w'], (1, 1), 'VALID') + params['bottom']['conv2']['b'])

    #------------------------------ expanding ------------------------------
    for y, layer_params in zip(ys, params['expanding']):
        x = expanding_layer(layer_params, x, y)

    # last conv layer reduces to 2 channels
    x = conv(x, params['final_layers']['conv1']['w'], (1, 1), 'VALID') + params['final_layers']['conv1']['b']

    #------------------------------ classification head ------------------------------
    # flatten
    x = jnp.reshape(x, (x.shape[0], -1))

    # linear
    x = activation(lax.dot(x, params['final_layers']['dense1']['w']) + params['final_layers']['dense1']['b'])
    x = activation(lax.dot(x, params['final_layers']['dense2']['w']) + params['final_layers']['dense2']['b'])
    x = nn.log_softmax(x)
    return x


def batched_loss_f(params, x, y_true):
    y_pred = batched_model(params, x)
    y_true_one_hot = nn.one_hot(y_true, num_classes = 2, dtype = int)
    return optax.softmax_cross_entropy(logits = y_pred, labels = y_true_one_hot).mean() # loss is already vectorized 


@jax.jit
def update(params, opt_state, x, y_true):
    loss, grads = value_and_grad(batched_loss_f, argnums = 0)(params, x, y_true)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state


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
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=img_dim, sample_fraction=0.02, augment=False)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=5, prefetch_factor=5, pin_memory=True)
    test_dl  = DataLoader(test_ds , batch_size=256, shuffle=True, drop_last=True, num_workers=1, prefetch_factor=1)

    # init
    initializer = nn.initializers.lecun_normal()
    params = init_params(initializer)
    optimizer = optax.lion(learning_rate = lr)
    opt_state = optimizer.init(params)
    activation = relu6

    num_params = format(sum(jax_array.size for jax_array in jax.tree_util.tree_flatten(params)[0]), ',')
    print(f'Learning rate {lr}')
    print(f'Batch size {batch_size}')
    print(f'Number of learnable parameters: {num_params}')
    print(f'Training on {len(train_ds)} examples and {len(train_dl)} batches of size {batch_size}')

    # train
    trained_params = train_and_eval(params, opt_state, train_dl, test_dl, num_epochs)