import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_simple_cnn(model_path, test_dir, img_size=(150, 150), out_dir='evaluation_outputs/simple_cnn'):
    os.makedirs(out_dir, exist_ok=True)

    print('Loading model:', model_path)
    model = tf.keras.models.load_model(model_path)

    # look folder for classes, one folder = one class
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    print('found classes ->', classes)

    y_true = []
    y_pred = []

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(test_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(cls_dir, fname)
            img = tf.keras.utils.load_img(path, target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr, verbose=0)
            pred = int(np.argmax(preds, axis=1)[0])
            y_true.append(idx)
            y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print('\nvery simple cnn check:')
    print('num samples =>', len(y_true))
    print('accuracy =>', f'{acc:.4f}')
    print('precision(w) =>', f'{precision:.4f}')
    print('recall(w) =>', f'{recall:.4f}')
    print('f1(w) =>', f'{f1:.4f}')
    print('\nconfusion matrix ->')
    print(cm)

    # save csv so later maybe use in slides
    cm_path = os.path.join(out_dir, 'simple_cnn_confusion_matrix.csv')
    np.savetxt(cm_path, cm, delimiter=',', fmt='%d')
    print('saved cm csv ->', cm_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple CNN evaluator')
    parser.add_argument('--model', required=True, help='Path to saved Keras model')
    parser.add_argument('--test_dir', required=True, help='Test images folder (one subfolder per class)')
    args = parser.parse_args()

    evaluate_simple_cnn(args.model, args.test_dir)
