import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

with open('mabel_predictions.pkl', 'rb') as f:
    data = pickle.load(f)

predictions = data['predictions']  # list of 3 arrays, each (n_test, num_classes)
y_test = data['y_test']
classes = data['classes']  # index -> family name

print(f"{len(classes)} families in play:")
for i, c in enumerate(classes):
    print(f"  {i}: {c}")

print()
names = ['paper (deep GBM)', 'shallow GBM', 'rf']
top_preds = []
for name, preds in zip(names, predictions):
    pred_labels = np.argmax(preds, axis=1)
    top_preds.append(pred_labels)
    acc = accuracy_score(y_test, pred_labels)
    print(f"{name}: accuracy = {acc:.4f}")

# How often do the 3 classifiers agree on their top prediction?
agree_all = (top_preds[0] == top_preds[1]) & (top_preds[1] == top_preds[2])
agree_two = ((top_preds[0] == top_preds[1]) | (top_preds[1] == top_preds[2]) | (top_preds[0] == top_preds[2])) & ~agree_all
agree_none = ~agree_all & ~agree_two
print()
print(f"All 3 agree: {agree_all.sum()} ({agree_all.mean():.1%})")
print(f"Exactly 2 agree: {agree_two.sum()} ({agree_two.mean():.1%})")
print(f"All disagree: {agree_none.sum()} ({agree_none.mean():.1%})")

# Confusion matrix for the best individual classifier, saved for inspection
best_idx = np.argmax([accuracy_score(y_test, tp) for tp in top_preds])
cm = confusion_matrix(y_test, top_preds[best_idx])
np.save('confusion_matrix_best.npy', cm)
print(f"\nSaved confusion matrix for best classifier ({names[best_idx]}) to confusion_matrix_best.npy")

# Top confused pairs (excluding the diagonal) -- this is the REAL data-driven check
# against the category groupings from earlier, not just reasoning about it abstractly.
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
flat_idx = np.argsort(cm_off_diag.ravel())[::-1][:15]
print("\nTop 15 most confused family PAIRS (true -> predicted):")
for idx in flat_idx:
    true_i, pred_i = np.unravel_index(idx, cm.shape)
    count = cm_off_diag[true_i, pred_i]
    if count > 0:
        print(f"  true={classes[true_i]:<25} predicted={classes[pred_i]:<25} count={count}")

with open('mabel_eval_results.pkl', 'wb') as f:
    pickle.dump({'top_preds': top_preds, 'y_test': y_test, 'classes': classes, 'names': names}, f)
print("\nSaved mabel_eval_results.pkl")
