# %% [markdown]
# # Model Training

# %%
from preprocessing import *
from model import get_model

# %%
backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# %%
model = get_model()
model.summary()

# %%
opt = SGD(learning_rate=0.01, momentum=0.94)
model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy', 'mse'])

# %%
history = model.fit(
    X_train,
    [y_train, y_train_boxes_scaled],
    epochs=10,
    batch_size=64,
    validation_split=0.1,
)

# %%
del X_train  # free memory after training
