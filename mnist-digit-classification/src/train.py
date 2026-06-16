# %% [markdown]
# # Model Training

# %%
from preprocessing import *
from model import build_cnn_model, build_ann_model, build_final_cnn

# %%
# Train CNN with BatchNorm
cnn_model = build_cnn_model()
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.summary()
model_history = cnn_model.fit(X_train_normalized, y_train_encoded, validation_split=0.2, epochs=5, verbose=2)

# %%
# Train ANN
ann_model = build_ann_model()
ann_model.summary()
history = ann_model.fit(
    X_train_normalized, y_train_encoded,
    epochs=15,
    validation_split=0.1,
    shuffle=True,
    batch_size=64,
    verbose=2,
)

# %%
# Train final CNN (best model)
model = build_final_cnn()
model.summary()
history_1 = model.fit(
    X_train_normalized, y_train_encoded,
    epochs=5,
    validation_split=0.1,
    shuffle=True,
    batch_size=64,
    verbose=2,
)
