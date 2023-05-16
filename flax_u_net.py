import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable



class Unet(nn.Module):
    model_depth: int
    img_dim: int
    activation: Callable = nn.relu6
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

        # can I convert to jax.lax.scan?
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
        x = nn.log_softmax(x)
        return x


batch_size  = 16
num_epochs  = 30
train_ratio = 0.8
lr = 2e-5
img_dim = 64

model = Unet(model_depth = 3, img_dim = img_dim)
params = model.init(jax.random.PRNGKey(0), jnp.ones((batch_size, img_dim, img_dim, 3)))
num_params = format(sum(jax_array.size for jax_array in jax.tree_util.tree_flatten(params)[0]), ',')

print(f'Number of parameters: {num_params}')


# where to jax.jit?
# how to create state

print()