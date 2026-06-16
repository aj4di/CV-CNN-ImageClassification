# %% [markdown]
# # Model Evaluation

# %%
from train import *

# %%
yhat1, yhat2 = model.predict(X_test)
yhat1_classes = np.argmax(yhat1, axis=1)

mae = mean_absolute_error(y_test_boxes_scaled, yhat2)
mse_val = mean_squared_error(y_test_boxes_scaled, yhat2)
acc = accuracy_score(y_test['Label'], yhat1_classes)

print('Bounding box MAE: %.3f' % mae)
print('Bounding box MSE: %.3f' % mse_val)
print('Label accuracy:   %.3f' % acc)

# %%
cf_matrix = confusion_matrix(y_test['Label'], yhat1_classes)
cf_matrix_n = cf_matrix / np.sum(cf_matrix, axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cf_matrix_n,
    xticklabels=np.unique(y_test['Label']),
    yticklabels=np.unique(y_test['Label']),
    annot=True,
)
plt.title('Normalized Confusion Matrix')
plt.show()
