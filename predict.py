import sys
import tensorflow as tf
import numpy as np

from extract import load_img
from model import get_model

def get_classifier_model():
    return tf.keras.models.load_model("model.h5")

if __name__ == "__main__":
    model = get_model()
    classifier_model = get_classifier_model()

    imgs = np.concatenate([load_img(x) for x in sys.argv[1:]], axis=0)
    embed = model.predict(imgs)
    allpred = classifier_model.predict(embed)
    classes = list(np.load("classes.npy"))

    for j in range(len(imgs)):
        print(f"{sys.argv[j + 1]}:")
        pred = [x[j] for x in allpred]

        for (idx, c) in enumerate(classes):
            cp = pred[idx].argmax() == 1
            if cp:
                print(f"{c}: yes ({pred[idx][1]})")
            elif False:
                print(f"{c}: no ({pred[idx][0]})")

        print("\n")
