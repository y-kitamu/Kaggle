import os
import csv

import chainer
from chainer.dataset.convert import concat_examples
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd


def evaluate(predictor, iterator, class_labels, output_stem, device=-1, meta_headers=["image_name"]):
    """Evaluate test data
    Args:
        predictor (chainer.Chain) : class instance that implement `predict` method
        iterator (chainer.dataset.iterator) :
        class_labels (list or string) :
        output_stem (string) : save output csv to `<output_stem>.csv`
        meta_headers (list of string) : header name of metadatas that write to output csv
    """
    if type(device) is int:
        device = chainer.cuda.get_device(device)

    if hasattr(iterator, "with_metadata"):
        iterator.with_metadata = True
    if hasattr(iterator, "is_one_hot"):
        iterator.is_one_hot = False

    output_dir = os.path.abspath(os.path.dirname(output_stem))
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataframes = []
    output_fname = "{}.csv".format(output_stem)
    fileobj = open(output_fname, "w")
    csv_writer = csv.writer(fileobj)
    csv_writer.writerow(meta_headers + ["true", "pred"] + class_labels)

    for batch in iterator:
        imgs, labels = concat_examples([b[:-1] for b in batch], device)
        with device:
            preds = predictor.predict(imgs)
            for pred, data in zip(preds, batch):
                pred_label = pred.argmax()
                csv_writer.writerow([data[-1][meta] for meta in meta_headers] + [data[1], pred_label] + pred.tolist())
    fileobj.close()
    df = pd.read_csv(output_fname)
    dataframes.append(df)
    show_metrics(df, class_labels)


def show_metrics(df, class_labels, true_header="true", pred_header="pred"):
    """Show accuracy, confusion_matrix, precision, recall, f1
    Args :
        df (pd.DataFrame) : cnn result dataframe
        class_labels (list of string) :
        true_header (string) : dataframe's header name of true label
        pred_header (string) : dataframe's header name of predict label
    """
    accuracy = accuracy_score(df[true_header], df[pred_header])
    labels = [i for i in range(len(class_labels))]
    cm = confusion_matrix(df[true_header], df[pred_header], labels=labels)
    print("Accuracy : {:.3f}".format(accuracy))
    print(cm)
    print(
        classification_report(df[true_header],
                              df[pred_header],
                              labels=range(len(class_labels)),
                              target_names=class_labels,
                              digits=3))
