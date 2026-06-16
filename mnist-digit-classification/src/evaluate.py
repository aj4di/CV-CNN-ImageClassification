# %% [markdown]
# # Model Evaluation

# %%
from train import *

# %%
# Evaluate ANN
accuracy_ann = ann_model.evaluate(X_test_normalized, y_test_encoded, verbose=2)
y_pred_ann = ann_model.predict(X_test_normalized)
y_pred_ann_arg = np.argmax(y_pred_ann, axis=1)
y_test_arg = np.argmax(y_test_encoded, axis=1)
cm_ann = tf.math.confusion_matrix(y_test_arg, y_pred_ann_arg)
print("ANN test accuracy:", accuracy_ann[1])

# %%
# Evaluate final CNN
accuracy_cnn = model.evaluate(X_test_normalized, y_test_encoded, verbose=2)
y_pred = model.predict(X_test_normalized)
y_pred_arg = np.argmax(y_pred, axis=1)
y_test_arg = np.argmax(y_test_encoded, axis=1)
cm_cnn = tf.math.confusion_matrix(y_test_arg, y_pred_arg)
print("CNN test accuracy:", accuracy_cnn[1])
