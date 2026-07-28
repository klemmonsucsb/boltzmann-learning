import pickle
import numpy as np
from sklearn.metrics import classification_report

with open('mabel_predictions_v2.pkl', 'rb') as f:
    data = pickle.load(f)

preds = data['predictions'][0]  # paper model, your best one
y_test = data['y_test']
classes = data['classes']
pred_labels = np.argmax(preds, axis=1)

report = classification_report(y_test, pred_labels, target_names=classes, output_dict=True, zero_division=0)
per_family = [(name, report[name]['f1-score'], report[name]['support']) for name in classes]
per_family.sort(key=lambda x: x[1])  # worst first

print("10 worst-performing families (name, f1-score, # test samples):")
for name, f1, support in per_family[:10]:
    print(f"  {name:<25} f1={f1:.3f}  n={int(support)}")
