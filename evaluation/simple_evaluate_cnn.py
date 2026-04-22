import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_simple_cnn(model_path, test_dir, img_size=(150, 150), out_dir='evaluation_outputs/simple_cnn'):
    # create out folder if not exist
    os.makedirs(out_dir, exist_ok=True)

    print('loading model ->', model_path)
    model = tf.keras.models.load_model(model_path)

    # look for subfolders = classes
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    print('classes ->', classes)

    y_true = []
    y_pred = []

    # very simple loop over files
    for i, cls in enumerate(classes):
        cls_path = os.path.join(test_dir, cls)
        for f in os.listdir(cls_path):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(cls_path, f)
            img = tf.keras.utils.load_img(p, target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
            arr = np.expand_dims(arr, 0)
            preds = model.predict(arr, verbose=0)
            pred_label = int(np.argmax(preds, axis=1)[0])
            y_true.append(i)
            y_pred.append(pred_label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0) if len(y_true) > 0 else (0.0, 0.0, 0.0, [])
    cm = confusion_matrix(y_true, y_pred) if len(y_true) > 0 else np.zeros((len(classes), len(classes)), dtype=int)

    # print simple results
    print('\nvery simple cnn results:')
    print('samples ->', len(y_true))
    print('accuracy ->', f'{acc:.4f}')
    print('precision(w) ->', f'{precision:.4f}')
    print('recall(w) ->', f'{recall:.4f}')
    print('f1(w) ->', f'{f1:.4f}')
    print('\nconfusion matrix:\n', cm)

    # save confusion matrix csv for slide later
    out_csv = os.path.join(out_dir, 'simple_cnn_confusion_matrix.csv')
    np.savetxt(out_csv, cm, delimiter=',', fmt='%d')
    print('saved cm csv ->', out_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Simple CNN evaluator')
    p.add_argument('--model', required=True, help='path to keras model')
    p.add_argument('--test_dir', required=True, help='test images dir (subfolders per class)')
    args = p.parse_args()
    evaluate_simple_cnn(args.model, args.test_dir)
