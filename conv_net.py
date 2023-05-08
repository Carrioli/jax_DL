import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, lax, nn, random, value_and_grad, vmap
from jax.nn import relu, relu6, selu
from optax import lion
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from load_data import LoadDataset


def init_params():
    master_key = random.PRNGKey(42)
    num_keys = 30
    keys = [random.PRNGKey(k) for k in random.randint(master_key, shape=(num_keys,), minval = -2**31, maxval = 2**31-1)]
    initializer = nn.initializers.lecun_normal()

    params = {
        'conv1': {
            'w': initializer(keys[0], (64, 3, 2, 2)), # out_c, in_c, h, w
            'b': initializer(keys[1], (64, 16, 16))
        },
        # 'norm1': {
        #     'gamma': jnp.ones((1, 64, 1, 1), dtype = jnp.float32), # good practice to init as 1.0
        #     'beta' : jnp.zeros((1, 64, 1, 1), dtype = jnp.float32) # good practice to init as 0.0
        # },
        'conv2': {
            'w': initializer(keys[2], (128, 64, 2, 2)),
            'b': initializer(keys[3], (128, 7, 7))
        },
        # 'norm2': {
        #     'gamma': jnp.ones((1, 128, 1, 1), dtype = jnp.float32),
        #     'beta' : jnp.zeros((1, 128, 1, 1), dtype = jnp.float32)
        # },
        'conv3': {
            'w': initializer(keys[4], (128, 128, 2, 2)),
            'b': initializer(keys[5], (128, 3, 3))
        },
        # 'norm3': {
        #     'gamma': jnp.ones((1, 128, 1, 1), dtype = jnp.float32),
        #     'beta' : jnp.zeros((1, 128, 1, 1), dtype = jnp.float32)
        # },
        'dense1': {
            'w': initializer(keys[6], (512, 512)),
            'b': initializer(keys[7], (512, 1)).squeeze() # initialized as 2D because of initializer, then squeeze
        },'dense2': {
            'w': initializer(keys[8], (512, 256)),
            'b': initializer(keys[9], (256, 1)).squeeze()
        },'dense3': {
            'w': initializer(keys[10], (256, 128)),
            'b': initializer(keys[11], (128, 1)).squeeze()
        },'dense4': {
            'w': initializer(keys[10], (128, 2)),
            'b': initializer(keys[11], (2, 1)).squeeze()
        }
    }
    return params


def batched_model(params, x):
    x = lax.conv(x, params['conv1']['w'], (2, 2), 'VALID') + params['conv1']['b']
    # x = lax.add(lax.mul(nn.standardize(x), params['norm1']['gamma']), params['norm1']['beta']) # pointwise mul and sum
    x = activation(x)
    x = hk.max_pool(value = x, window_shape = (2, 2), strides=(1, 1), padding='VALID')

    x = lax.conv(x, params['conv2']['w'], (2, 2), 'VALID') + params['conv2']['b']
    # x = lax.add(lax.mul(nn.standardize(x), params['norm2']['gamma']), params['norm2']['beta'])
    x = activation(x)
    x = hk.max_pool(value = x, window_shape = (2, 2), strides=(1, 1), padding='VALID')

    x = lax.conv(x, params['conv3']['w'], (2, 2), 'VALID') + params['conv3']['b']
    # x = lax.add(lax.mul(nn.standardize(x), params['norm3']['gamma']), params['norm3']['beta'])
    x = activation(x)
    x = hk.max_pool(value = x, window_shape = (2, 2), strides=(1, 1), padding='VALID')
    
    x = jnp.reshape(x, (x.shape[0], -1)) # flatten layer, 2D output

    x = x @ params['dense1']['w'] + params['dense1']['b']
    x = activation(x)

    x = x @ params['dense2']['w'] + params['dense2']['b']
    x = activation(x)

    x = x @ params['dense3']['w'] + params['dense3']['b']
    x = activation(x)

    x = x @ params['dense4']['w'] + params['dense4']['b']
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
    batch_size  = 200
    num_epochs  = 30
    train_ratio = 0.8

    # get data
    sub_path = './'
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=32, sample_fraction=1.0, augment=True)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, prefetch_factor=15)
    test_dl  = DataLoader(test_ds , batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, prefetch_factor=15)


    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=1.0,
    #     warmup_steps=50,
    #     decay_steps=1_000,
    #     end_value=0.0,
    # )

    # optimizer = optax.chain(
    #     optax.clip(1.0),
    #     optax.adamw(learning_rate=schedule),
    # )

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
    bar.colour = '#0000ff'
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