from sys import exit

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, nn, random, value_and_grad, vmap
from jax.lax import conv, conv_transpose
from jax.nn import relu, relu6, selu
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from load_data import LoadDataset


# constants
batch_size  = 16
num_epochs  = 30
train_ratio = 0.8
lr = 2e-5
img_dim = 64
model_depth = 3


def init_params(initializer):
    master_key = random.PRNGKey(0)
    num_keys = 100
    keys = random.split(master_key, num=num_keys)

    # width and height at the bottom of the U
    bottom_dim = (img_dim - 2 ** (model_depth + 2) + 4) // 2 ** (model_depth - 1)

    # Note contracting and expanding are lists because we will use lax.scan()
    # Every contracting layer has 2 conv layers
    params = {}
    params['contracting'] = [
        {
            'conv1': {
                'w': initializer(
                    keys[4 * i], (2 ** (i + 6), 3 if i == 0 else 2 ** (i + 5), 3, 3)
                ),  # out_c, in_c, h, w
                'b': initializer(
                    keys[4 * i + 1],
                    (
                        2 ** (i + 6),
                        (img_dim - 6 * 2 ** i + 4) // 2 ** i,
                        (img_dim - 6 * 2 ** i + 4) // 2 ** i,
                    ),
                ),
            },
            'conv2': {
                'w': initializer(keys[4 * i + 2], (2 ** (i + 6), 2 ** (i + 6), 3, 3)),
                'b': initializer(
                    keys[4 * i + 3],
                    (
                        2 ** (i + 6),
                        (img_dim - 2 ** (i + 3) + 4) // 2 ** i,
                        (img_dim - 2 ** (i + 3) + 4) // 2 ** i,
                    ),
                ),
            },
        }
        for i in range(model_depth - 1)
    ]

    i = model_depth - 1
    params['bottom'] = {
        'conv1': {
            'w': initializer(
                keys[4 * i], (2 ** (i + 6), 3 if i == 0 else 2 ** (i + 5), 3, 3)
            ),  # out_c, in_c, h, w
            'b': initializer(
                keys[4 * i + 1],
                (
                    2 ** (i + 6),
                    (img_dim - 6 * 2 ** i + 4) // 2 ** i,
                    (img_dim - 6 * 2 ** i + 4) // 2 ** i,
                ),
            ),
        },
        'conv2': {
            'w': initializer(keys[4 * i + 2], (2 ** (i + 6), 2 ** (i + 6), 3, 3)),
            'b': initializer(
                keys[4 * i + 3],
                (
                    2 ** (i + 6),
                    (img_dim - 2 ** (i + 3) + 4) // 2 ** i,
                    (img_dim - 2 ** (i + 3) + 4) // 2 ** i,
                ),
            ),
        },
    }

    # Every expanding layer has 3 conv layers
    params['expanding'] = [
        {
            'conv1': {
                'w': initializer(
                    keys[model_depth + 4 * i],
                    (2 ** (model_depth + 4 - i), 2 ** (model_depth + 5 - i), 2, 2),
                ),
                'b': initializer(
                    keys[model_depth + 4 * i + 1],
                    (
                        2 ** (model_depth + 4 - i),
                        2 ** i * (2 * bottom_dim - 8) + 8,
                        2 ** i * (2 * bottom_dim - 8) + 8,
                    ),
                ),
            },
            'conv2': {
                'w': initializer(
                    keys[model_depth + 4 * i + 2],
                    (2 ** (model_depth + 4 - i), 2 ** (model_depth + 5 - i), 3, 3),
                ),
                'b': initializer(
                    keys[model_depth + 4 * i + 3],
                    (
                        2 ** (model_depth + 4 - i),
                        2 ** i * (2 * bottom_dim - 8) + 6,
                        2 ** i * (2 * bottom_dim - 8) + 6,
                    ),
                ),
            },
            'conv3': {
                'w': initializer(
                    keys[model_depth + 4 * i + 4],
                    (2 ** (model_depth + 4 - i), 2 ** (model_depth + 4 - i), 3, 3),
                ),
                'b': initializer(
                    keys[model_depth + 4 * i + 5],
                    (
                        2 ** (model_depth + 4 - i),
                        2 ** i * (2 * bottom_dim - 8) + 4,
                        2 ** i * (2 * bottom_dim - 8) + 4,
                    ),
                ),
            },
        }
        for i in range(model_depth - 1)
    ]

    i = model_depth - 2
    params['final_layers'] = {
        'conv1': {
            'w': initializer(
                keys[model_depth + 4 * i + 6],
                (2 ** (model_depth + 4 - i), 2 ** (model_depth + 4 - i), 3, 3),
            ),
            'b': initializer(
                keys[model_depth + 4 * i + 7],
                (
                    2 ** (model_depth + 4 - i),
                    2 ** i * (2 * bottom_dim() - 8) + 4,
                    2 ** i * (2 * bottom_dim() - 8) + 4,
                ),
            ),
        },
        'dense1': {
            'w': initializer(
                keys[model_depth + 4 * i + 8],
                (2 ** (model_depth + 4 - i) * (2 * bottom_dim() - 8) ** 2, 512),
            ),
            'b': initializer(keys[model_depth + 4 * i + 9], (512, 1)).squeeze()
        },
        'dense2': {
            'w': initializer(keys[model_depth + 4 * i + 10], (512, 2)),
            'b': initializer(keys[model_depth + 4 * i + 11], (2, 1)).squeeze()
        }
    }

    return params


def batched_model(params, x):

    #------------------------------ contracting ------------------------------

    x = activation(conv(x, params['conv1']['w'], (1, 1), 'VALID') + params['conv1']['b'])
    x = activation(conv(x, params['conv2']['w'], (1, 1), 'VALID') + params['conv2']['b'])
    y = jnp.copy(x)
    x = hk.max_pool(value = x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    x = activation(conv(x, params['conv3']['w'], (1, 1), 'VALID') + params['conv3']['b'])
    x = activation(conv(x, params['conv4']['w'], (1, 1), 'VALID') + params['conv4']['b'])
    z = jnp.copy(x)
    x = hk.max_pool(value = x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

    # bottom layers
    x = activation(conv(x, params['conv5']['w'], (1, 1), 'VALID') + params['conv5']['b'])
    x = activation(conv(x, params['conv6']['w'], (1, 1), 'VALID') + params['conv6']['b'])


    #------------------------------ expanding ------------------------------

    # Expansion 1
    # Upsample
    x = conv_transpose(x, params['conv7']['w'], (2, 2), 'VALID', dimension_numbers=('NCHW', 'OIHW', 'NCHW')) + params['conv7']['b']
    # Concatenate
    start = (z.shape[-1] - x.shape[-1]) // 2
    end   = start + x.shape[-1]
    x = jnp.concatenate((x, z[:,:,start:end,start:end]), axis = 1)
    # 2, 3x3 convolutions + activation
    x = activation(conv(x, params['conv8']['w'], (1, 1), 'VALID') + params['conv8']['b'])
    x = activation(conv(x, params['conv9']['w'], (1, 1), 'VALID') + params['conv9']['b'])
    
    # Expansion 2
    x = conv_transpose(x, params['conv10']['w'], (2, 2), 'VALID', dimension_numbers=('NCHW', 'OIHW', 'NCHW')) + params['conv10']['b']
    start = (y.shape[-1] - x.shape[-1]) // 2
    end   = start + x.shape[-1]
    x = jnp.concatenate((x, y[:,:,start:end,start:end]), axis = 1)
    x = activation(conv(x, params['conv11']['w'], (1, 1), 'VALID') + params['conv11']['b'])
    x = activation(conv(x, params['conv12']['w'], (1, 1), 'VALID') + params['conv12']['b'])

    # last conv layer
    x = conv(x, params['conv13']['w'], (1, 1), 'VALID') + params['conv13']['b']

    # flatten
    x = jnp.reshape(x, (x.shape[0], -1))

    # linear
    x = relu(x @ params['dense1']['w'] + params['dense1']['b'])
    x = x @ params['dense2']['w'] + params['dense2']['b']
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
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=img_dim, sample_fraction=0.2, augment=False)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6, prefetch_factor=5, pin_memory=True)
    test_dl  = DataLoader(test_ds , batch_size=256, shuffle=True, drop_last=True, num_workers=6, prefetch_factor=5)

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