import os
import numpy as np
import csv

from src.dataset import TestDatasetGenerator
from src.model import get_model


def predict(cfg,
            output_filename="./submission.csv",
            model_dir=os.path.join(os.path.dirname(__file__), "../results/baseline/"),
            test_data_dir="../input/cassava-leaf-disease-classification"):
    test_ds = TestDatasetGenerator(test_data_dir)
    model = get_model(cfg)
    preds = np.zeros((len(test_ds), cfg["n_classes"]))
    for metrix in cfg["test"]["models"]:
        for idx in cfg["train"]["k_fold"]:
            model_filename = os.path.join(model_dir, "{}{}.hdf5".format(metrix, idx))
            print("Start evaluate using {}".format(model_filename))
            model.load_weight(model_filename)
            results = []
            for imgs in test_ds:
                res = model(imgs).numpy()
                results.append(res)
            preds += np.concatenate(results, axis=0)
    preds = preds.argmax(axis=1)
    with open(output_filename, 'w') as f:
        csv_writer = csv.writer(f, lineterminator="\n")
        csv_writer.writerow("image_id", "pred")
        for fname, pred in zip(test_ds.filenames, preds):
            csv_writer.writerow([fname, pred])
    return preds
