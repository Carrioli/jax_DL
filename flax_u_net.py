from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import lax, random, value_and_grad
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from load_data import LoadDataset


class Unet(nn.Module):
    model_depth: int
    img_dim: int
    activation: Callable = nn.activation.relu6
    num_classes: int = 2

    def setup(self):
        # contracting path
        self.contracting_convs = [(nn.Conv(2 ** (i + 6), (3, 3), padding='VALID'),
                                   nn.Conv(2 ** (i + 6), (3, 3), padding='VALID'))
                                   for i in range(self.model_depth - 1)]

        # bottom
        self.bottom_conv1 = nn.Conv(2 ** (self.model_depth + 5), (3, 3), padding='VALID')
        self.bottom_conv2 = nn.Conv(2 ** (self.model_depth + 5), (3, 3), padding='VALID')

        # expanding path
        self.expanding_convs = [(nn.ConvTranspose(2 ** (self.model_depth + 4 - i), kernel_size=(2, 2), strides=(2, 2), padding='VALID'),
                                 nn.Conv(2 ** (self.model_depth + 4 - i), (3, 3), padding='VALID'),
                                 nn.Conv(2 ** (self.model_depth + 4 - i), (3, 3), padding='VALID'))
                                 for i in range(self.model_depth - 1)]
        
        # classification head
        self.last_conv = nn.Conv(2, (1, 1), padding='VALID')
        self.dense1 = nn.Dense(512)
        self.dense2 = nn.Dense(self.num_classes)


    def __call__(self, x):
        ys = [None] * (self.model_depth - 1)

        for i, (conv1, conv2) in enumerate(self.contracting_convs):
            x = self.activation(conv1(x))
            x = self.activation(conv2(x))
            ys[self.model_depth - 2 - i] = jnp.copy(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        
        x = self.bottom_conv1(x)
        x = self.bottom_conv2(x)

        for i, (conv_trans, conv1, conv2) in enumerate(self.expanding_convs):
            y = ys[i]
            x = conv_trans(x)
            start = (y.shape[1] - x.shape[1]) // 2
            end   = start + x.shape[1]
            x = jnp.concatenate((x, y[:,start:end,start:end,:]), axis = -1)
            x = self.activation(conv1(x))
            x = self.activation(conv2(x))
        
        x = self.last_conv(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = nn.activation.log_softmax(x)
        return x


def batched_loss_f(params, x, y_true):
    y_pred = model.apply(params, x)
    y_true_one_hot = nn.one_hot(y_true, num_classes = 2, dtype = int)
    return optax.softmax_cross_entropy(logits = y_pred, labels = y_true_one_hot).mean() # loss is already vectorized 


@jax.jit
def update(state, x, y_true):
    loss, grads = value_and_grad(batched_loss_f, argnums = 0)(state.params, x, y_true)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def eval(state, x_batch, y_batch):
    y_pred_batch = model.apply(state.params, x_batch)
    accuracy = jnp.mean(jnp.argmax(y_pred_batch, -1) == y_batch)
    return accuracy


def eval_fn(state, data_loader):
    accuracies = [eval(state, jnp.array(x_batch), jnp.array(y_batch)) for (x_batch, y_batch) in iter(data_loader)]
    return sum(accuracies)/len(accuracies)


def train_epoch(state, data_loader, bar):
    epoch_loss = 0.0
    for (batch_img, batch_label) in iter(data_loader):
        x = jnp.array(batch_img) # batches need to be converted to jnp array
        y_true = jnp.array(batch_label)
        state, batch_loss = update(state, x, y_true)
        epoch_loss += batch_loss
        bar.update(1)
    return state, epoch_loss


def train_and_eval(state, train_dl, test_dl, n_epochs):
    bar = tqdm(total = n_epochs * len(train_dl), ncols = 150, leave = True)
    bar.colour = '#0000ff'
    for epoch in range(n_epochs):
        bar.set_postfix(epoch = f'{epoch + 1} out of {n_epochs}')
        state, epoch_loss = train_epoch(state, train_dl, bar)
        # eval for each epoch
        accuracy = eval_fn(state, test_dl)
        tqdm.write(f'Epoch: {epoch + 1}, average epoch loss: {epoch_loss/len(train_dl):.6f}. Test accuracy: {accuracy}')
    return state.params



if __name__ == '__main__':
    # constants
    batch_size  = 16
    num_epochs  = 30
    train_ratio = 0.8
    lr = 2e-5
    img_dim = 64

    # get data
    sub_path = 'datasets/histopathologic-cancer-detection/'
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=img_dim, sample_fraction=0.02, augment=False, channels_last=True)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=5, prefetch_factor=5, pin_memory=True)
    test_dl  = DataLoader(test_ds , batch_size=256, shuffle=True, drop_last=True, num_workers=1, prefetch_factor=1)


    # init model and state
    model = Unet(model_depth = 3, img_dim = img_dim)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), jnp.ones((batch_size, img_dim, img_dim, 3))),
        tx=optax.lion(learning_rate = lr)
    )

    num_params = format(sum(jax_array.size for jax_array in jax.tree_util.tree_flatten(state.params)[0]), ',')
    print(f'Learning rate {lr}')
    print(f'Batch size {batch_size}')
    print(f'Number of learnable parameters: {num_params}')
    print(f'Training on {len(train_ds)} examples and {len(train_dl)} batches of size {batch_size}')

    trained_params = train_and_eval(state, train_dl, test_dl, num_epochs)
