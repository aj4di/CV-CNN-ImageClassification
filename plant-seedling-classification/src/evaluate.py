# %% [markdown]
# # Model Evaluation

# %%
from train import *


def evaluate_model(model, X_test_norm, y_test_enc, name=""):
    """Evaluate model and return confusion matrix."""
    acc = model.evaluate(X_test_norm, y_test_enc, verbose=2)
    print(f"{name} test accuracy: {acc[1]:.4f}")

    y_pred = model.predict(X_test_norm)
    y_pred_arg = np.argmax(y_pred, axis=1)
    y_test_arg = np.argmax(y_test_enc, axis=1)
    cm = tf.math.confusion_matrix(y_test_arg, y_pred_arg)
    return cm


# %%
cm1 = evaluate_model(model1, X_test_normalized, y_test_encoded, "Model 1 (Simple CNN)")
cm2 = evaluate_model(model2, X_test_normalized, y_test_encoded, "Model 2 (CNN+Aug+BN)")
cm_vgg = evaluate_model(model_vgg, X_test_normalized, y_test_encoded, "Model 3 (VGG16)")
cm_xcep = evaluate_model(xception_model, X_test_normalized2, y_test_encoded2, "Model 4 (Xception)")

# %%
# Sample predictions using LabelBinarizer inverse_transform
sample_idx = 2
print('Predicted:', enc.inverse_transform(model2.predict(X_test_normalized[sample_idx].reshape(1, 64, 64, 3))))
print('True:     ', enc.inverse_transform(y_test_encoded)[sample_idx])
