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


class ViT():
    def __init__(self, img_dim, patch_size, num_layers, num_heads, initializer, optimizer):
        self.img_dim = img_dim # static
        self.patch_size = patch_size # static
        self.num_layers = num_layers # static
        self.num_heads = num_heads # static
        self.initializer = initializer # static
        self.optimizer = optimizer # static 
        self.num_tokens = (img_dim // patch_size) ** 2 + 1 # static
        self.D = (self.patch_size ** 2) * 3 # static
        self.d_k = self.D // num_heads # static
        self.d_v = self.D // num_heads # static
        self.params = self.init_params() # dynamic
        self.opt_state = optimizer.init(self.params) # dynamic


    def __repr__(self) -> str:
        num_params = format(sum(jax_array.size for jax_array in jax.tree_util.tree_flatten(self.params)[0]), ',')
        return f"""\
            \nThis is a ViT model with {num_params} parameters.\
            \nImage dimension: {self.img_dim}.\
            \nPatch size: {self.patch_size}.\
            \nNumber of tokens (ie patches + cls token) is {self.num_tokens}.\
            \nNumber of transformer encoder layers: {self.num_layers}, each with {self.num_heads} heads.
        """
    
    def tree_flatten(self):
        static_data = (self.img_dim, self.patch_size, self.num_layers, self.num_heads, self.initializer, self.optimizer)
        dynamic_data = (self.params, self.opt_state)
        return dynamic_data, static_data

    @classmethod
    def tree_unflatten(cls, static_data, dynamic_data):
        img_dim, patch_size, num_layers, num_heads, initializer, optimizer = static_data
        params, opt_state = dynamic_data
        instance = cls(img_dim, patch_size, num_layers, num_heads, initializer, optimizer)
        instance.params = params
        instance.opt_state = opt_state
        return instance



    def init_params(self):
        master_key = random.PRNGKey(42)
        num_keys = 30
        keys = [random.PRNGKey(k) for k in random.randint(master_key, shape=(num_keys,), minval = -2**31, maxval = 2**31-1)]
        params = {
            'dense1': {
                'w': self.initializer(keys[0], (self.D, self.D)),
                'b': self.initializer(keys[1], (self.D, 1)).squeeze() # initialized as 2D because of initializer, then squeeze
            },
            'cls_token': self.initializer(keys[2], (self.D, 1)).squeeze(),
            'position_embeddings': self.initializer(keys[3], (self.num_tokens, self.D)),
            'transformer_block': {
                'W_Q_all':  self.initializer(keys[3], (self.num_layers, self.num_heads, self.D, self.d_k)),
                'W_K_all':  self.initializer(keys[4], (self.num_layers, self.num_heads, self.D, self.d_k)),
                'W_V_all':  self.initializer(keys[5], (self.num_layers, self.num_heads, self.D, self.d_v)),
                'W_O':      self.initializer(keys[6], (self.num_layers, self.num_heads * self.d_v, self.D)), # final linear layer
                'mlp_W1':   self.initializer(keys[7], (self.num_layers, self.D, 4 * self.D)),
                'mlp_W2':   self.initializer(keys[8], (self.num_layers, 4 * self.D, self.D)),
            }
        }
        return params


    def extract_patches(self, x):
        C, H, W = x.shape
        x = x.reshape(C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.transpose(1, 3, 0, 2, 4)
        patches = x.reshape(-1, C * self.patch_size * self.patch_size)
        return patches


    def singlehead_self_attention(self, x, W_Q, W_K, W_V):
        Q = jnp.matmul(x, W_Q)
        K = jnp.matmul(x, W_K)
        V = jnp.matmul(x, W_V)
        d_k = W_Q.shape[-1]
        scores = jnp.matmul(Q, jnp.transpose(K)) / jnp.sqrt(d_k)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(attention_weights, V)


    def multihead_self_attention(self, x, W_Q, W_K, W_V, W_O):
        # get all the heades, concatenate them, and then apply the final linear layer
        x = jax.vmap(self.singlehead_self_attention, in_axes=[None, 0, 0, 0])(x, W_Q, W_K, W_V)
        # concatenate the heads so that x has shape (x.shape[0]*x.shape[1], x.shape[2])
        x = x.reshape(x.shape[1], -1)
        return jnp.matmul(x, W_O)


    def mlp(self, x, W1, W2):
        return gelu(jnp.matmul(gelu(jnp.matmul(x, W1)), W2))


    def transformer_block(self, z_l_minus_1, layer_idx):
        W_Q    = self.params['transformer_block']['W_Q_all'][layer_idx]
        W_K    = self.params['transformer_block']['W_K_all'][layer_idx]
        W_V    = self.params['transformer_block']['W_V_all'][layer_idx]
        W_O    = self.params['transformer_block']['W_O'][layer_idx]
        mlp_W1 = self.params['transformer_block']['mlp_W1'][layer_idx]
        mlp_W2 = self.params['transformer_block']['mlp_W2'][layer_idx]

        z_prime_l = self.multihead_self_attention(standardize(z_l_minus_1, axis=1), W_Q, W_K, W_V, W_O) + z_l_minus_1
        z_l = self.mlp(standardize(z_prime_l, axis=1), mlp_W1, mlp_W2) + z_prime_l
        return z_l, None


    def model(self, x):
        x = self.extract_patches(x)
        x = x @ self.params['dense1']['w'] + self.params['dense1']['b']
        # Tokenization: Add the classification token
        x = jnp.concatenate((self.params['cls_token'][jnp.newaxis, :], x), axis=0)
        # Add learable positional encoding
        x += self.params['position_embeddings'] # save for later

        _, final_state = jax.lax.scan(self.transformer_block, x, jnp.arange(self.num_layers), self.num_layers)

        return x


    def batched_model(self, x):
        return jax.vmap(self.model, in_axes=(0))(x)


    def batched_loss_f(self, x, y_true):
        y_pred = self.batched_model(x)
        y_true_one_hot = nn.one_hot(y_true, num_classes = 2, dtype = int)
        return optax.softmax_cross_entropy(logits = y_pred, labels = y_true_one_hot).mean() # loss is already vectorized 


    @jax.jit
    def update(self, x, y_true):
        loss, grads = value_and_grad(self.batched_loss_f, argnums = 0)(x, y_true)
        updates, self.opt_state = optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return loss


    @jax.jit
    def eval(self, params, x_batch, y_batch):
        y_pred_batch = self.batched_model(params, x_batch)
        accuracy = jnp.mean(jnp.argmax(y_pred_batch, -1) == y_batch)
        return accuracy


    def train_epoch(self, data_loader, bar):
        epoch_loss = 0.0
        for (batch_img, batch_label) in iter(data_loader):
            batch_loss, self.opt_state = self.update(jnp.array(batch_img), jnp.array(batch_label)) # batches need to be converted to jnp array
            epoch_loss += batch_loss
            bar.update(1)
        return epoch_loss


    def train_and_eval(self, num_epochs, train_dl, test_dl):
        bar = tqdm(total = num_epochs * len(train_dl), ncols = 150, leave = True)
        bar.colour = '#0000ff'
        for epoch in range(num_epochs):
            bar.set_postfix(epoch = f'{epoch + 1} out of {num_epochs}')
            epoch_loss = self.train_epoch(train_dl, bar)
            # eval for each epoch
            accuracies = [self.eval(params, jnp.array(x_batch), jnp.array(y_batch)) for (x_batch, y_batch) in iter(test_dl)]
            tqdm.write(f'Epoch: {epoch + 1}, average epoch loss: {epoch_loss/len(train_dl):.6f}. Test accuracy: {sum(accuracies)/len(accuracies)}')


if __name__ == '__main__':
    # constants
    batch_size  = 200
    train_ratio = 0.8
    img_dim = 96

    # get data
    sub_path = './'
    ds = LoadDataset(sub_path + 'train', sub_path + 'train_labels.csv', dimension=img_dim, sample_fraction=0.1, augment=False)
    num_train = int(train_ratio*len(ds))
    num_test = len(ds) - num_train
    train_ds, test_ds = random_split(ds, [num_train, num_test])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, prefetch_factor=1)
    test_dl  = DataLoader(test_ds , batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, prefetch_factor=1)

    print(f'Batch size {batch_size}')
    print(f'Training on {len(train_ds)} examples and {len(train_dl)} batches of size {batch_size}')

    jax.tree_util.register_pytree_node(
        ViT,
        ViT.tree_flatten,
        ViT.tree_unflatten
    )

    # init model
    vision_transformer = ViT(img_dim = img_dim, patch_size = 16, num_layers = 8, num_heads = 8, initializer=nn.initializers.lecun_normal(), optimizer = optax.lion(learning_rate = 2e-5))
    print(vision_transformer)

    vision_transformer.train_and_eval(num_epochs = 10, train_dl = train_dl, test_dl = test_dl)