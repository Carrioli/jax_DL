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


def init_params():
    master_key = random.PRNGKey(42)
    num_keys = 50
    keys = [random.PRNGKey(k) for k in random.randint(master_key, shape=(num_keys,), minval = -2**31, maxval = 2**31-1)]
    initializer = nn.initializers.lecun_normal()

    params = {
        # contracting
        'conv1': {
            'w': initializer(keys[0], (64, 3, 3, 3)), # out_c, in_c, h, w
            'b': initializer(keys[1], (64, 62, 62))
        },'conv2': {
            'w': initializer(keys[2], (64, 64, 3, 3)),
            'b': initializer(keys[3], (64, 60, 60))
        },'conv3': {
            'w': initializer(keys[4], (128, 64, 3, 3)),
            'b': initializer(keys[5], (128, 28, 28))
        },'conv4': {
            'w': initializer(keys[6], (128, 128, 3, 3)),
            'b': initializer(keys[7], (128, 26, 26))
        },'conv5': {
            'w': initializer(keys[8], (256, 128, 3, 3)),
            'b': initializer(keys[9], (256, 11, 11))
        },'conv6': {
            'w': initializer(keys[10], (256, 256, 3, 3)),
            'b': initializer(keys[11], (256, 9, 9))
        },
        # expanding
        'conv7': {
            'w': initializer(keys[12], (128, 256, 2, 2)),
            'b': initializer(keys[13], (128, 18, 18))
        },'conv8': {
            'w': initializer(keys[14], (128, 256, 3, 3)),
            'b': initializer(keys[15], (128, 16, 16))
        },'conv9': {
            'w': initializer(keys[16], (128, 128, 3, 3)),
            'b': initializer(keys[17], (128, 14, 14))
        },'conv10': {
            'w': initializer(keys[18], (64, 128, 2, 2)),
            'b': initializer(keys[19], (64, 28, 28))
        },'conv11': {
            'w': initializer(keys[20], (64, 128, 3, 3)),
            'b': initializer(keys[21], (64, 26, 26))
        },'conv12': {
            'w': initializer(keys[22], (64, 64, 3, 3)),
            'b': initializer(keys[23], (64, 24, 24))
        },
        # last conv layer
        'conv13': {
            'w': initializer(keys[24], (2, 64, 1, 1)),
            'b': initializer(keys[25], (2, 24, 24))
        },'dense1': {
            'w': initializer(keys[26], (1152, 512)),
            'b': initializer(keys[27], (512, 1)).squeeze()
        },'dense2': {
            'w': initializer(keys[28], (512, 2)),
            'b': initializer(keys[29], (2, 1)).squeeze()
        }
    }
    return params


def batched_model(params, x):

    #------------------------------ contracting ------------------------------
        
    x = activation(conv(x, params['conv1']['w'], (1, 1), 'VALID') + params['conv1']['b'])
    x = activation(conv(x, params['conv2']['w'], (1, 1), 'VALID') + params['conv2']['b'])
    y = jnp.copy(x)
    x = hk.max_pool(value = x, window_shape = (2, 2), strides=(2, 2), padding='VALID')

    x = activation(conv(x, params['conv3']['w'], (1, 1), 'VALID') + params['conv3']['b'])
    x = activation(conv(x, params['conv4']['w'], (1, 1), 'VALID') + params['conv4']['b'])
    z = jnp.copy(x)
    x = hk.max_pool(value = x, window_shape = (2, 2), strides=(2, 2), padding='VALID')

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


if __name__ == '__main__':
    # constants
    batch_size  = 16
    num_epochs  = 30
    train_ratio = 0.8

    # get data
    # sub_path = 'histopathologic-cancer-detection/'
    sub_path = './'
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=64, sample_fraction=1.0, augment=True)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=30, prefetch_factor=30, pin_memory=True)
    test_dl  = DataLoader(test_ds , batch_size=256, shuffle=True, drop_last=True, num_workers=30, prefetch_factor=30)

    # init
    params = init_params()
    lr = 2e-5
    optimizer = optax.lion(learning_rate = lr)
    opt_state = optimizer.init(params)
    activation = relu6

    num_params = format(sum(jax_array.size for jax_array in jax.tree_util.tree_flatten(params)[0]), ',')
    print(f'Learning rate {lr}')
    print(f'Batch size {batch_size}')
    print(f'Number of parametes: {num_params}')
    print(f'Training on {len(train_ds)} examples and {len(train_dl)} batches of size {batch_size}')

    # train
    bar = tqdm(total = num_epochs * len(train_dl), ncols = 150, leave = True)
    bar.colour = '0000ff'
    for epoch in range(num_epochs):
        bar.set_postfix(epoch = f'{epoch + 1} out of {num_epochs}')
        epoch_loss = 0.0
        for (batch_img, batch_label) in iter(train_dl):
            batch_loss, params, opt_state = update(params, opt_state, jnp.array(batch_img), jnp.array(batch_label)) # batches need to be converted to jnp array
            epoch_loss += batch_loss
            bar.update(1)
        # eval for each epoch
        accuracies = [eval(params, jnp.array(x_batch), jnp.array(y_batch)) for (x_batch, y_batch) in iter(test_dl)]
        tqdm.write(f'Epoch: {epoch + 1}, average epoch loss: {epoch_loss/len(train_dl):.6f}. Test accuracy: {sum(accuracies)/len(accuracies)}')