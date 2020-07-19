import os
import csv

import chainer
from chainer.dataset.convert import concat_examples
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import pandas as pd

from melanoma import constants


def evaluate_ensemble(predictor,
                      iterator,
                      class_labels,
                      output_stem,
                      npz_files,
                      device=-1,
                      meta_header=["image_name"]):
    dataframes = []
    for npz_file in npz_files:
        chainer.serializers.load_npz(npz_file, predictor.extractor)
        footer = npz_file.split("/")[-2]
        output = f"{output_stem}_{footer}"
        dataframes.append(evaluate(
            predictor,
            iterator,
            class_labels,
            output,
            device,
        ))
    print("\nEnsemble Result : \n")
    df = pd.concat(dataframes)
    show_metrics(df, class_labels)


def evaluate(predictor, iterator, class_labels, output_stem, device=-1, meta_headers=["image_name"]):
    """Evaluate test data
    Args:
        predictor (chainer.Chain) : class instance that implement `predict` method
        iterator (chainer.dataset.iterator) :
        class_labels (list or string) :
        output_stem (string) : save output csv to `<output_stem>.csv`
        meta_headers (list of string) : header name of metadatas that write to output csv
    """
    if device >= 0:
        predictor.to_gpu(device)
        predictor.extractor.to_gpu(device)

    if type(device) is int:
        device = chainer.cuda.get_device(device)

    if hasattr(iterator.dataset, "with_metadata"):
        iterator.dataset.with_metadata = True

    output_dir = os.path.abspath(os.path.dirname(output_stem))
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_fname = "{}.csv".format(output_stem)
    fileobj = open(output_fname, "w")
    csv_writer = csv.writer(fileobj)
    csv_writer.writerow(meta_headers + ["true", "pred"] + ["conf_{}".format(label) for label in class_labels])

    iterator.reset()
    for batch in iterator:
        inputs = concat_examples([b[:-1] for b in batch], device)
        with device:
            preds = predictor.predict(*inputs)
            for pred, data, label in zip(preds, batch, inputs[-1]):
                if pred.shape[-1] > 1:
                    pred_label = pred.argmax(axis=-1)
                    confs = pred.tolist()
                else:
                    pred_label = int(pred.round())
                    confs = [pred]
                if label.shape[-1] > 1:
                    label = label.argmax(axis=-1)
                csv_writer.writerow([data[-1][meta] for meta in meta_headers] + [label, pred_label] + confs)
    fileobj.close()
    df = pd.read_csv(output_fname)
    show_metrics(df, class_labels)
    return df


def evaluate_submission(predictor, iterator, output_stem, device, filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    output_list = []
    for idx, filename in enumerate(sorted(filenames)):
        chainer.serializers.load_npz(filename, predictor.extractor)
        if device >= 0:
            predictor.to_gpu(device)
            predictor.extractor.to_gpu(device)
        output_list.append(_evaluate_submission(predictor, iterator, f"{output_stem}_{idx:02d}", device))

    _sum_predict(output_list, output_stem)


def _sum_predict(output_list, output_stem):
    image_preds = {}
    for fname in output_list:
        print(fname)
        with open(fname, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            for row in csv_reader:
                if row[0] not in image_preds:
                    image_preds[row[0]] = []
                image_preds[row[0]].append(row[1])

    output_filename = f"{output_stem}.csv"
    with open(output_filename, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["image_name", "target"])
        for key, val in image_preds.items():
            csv_writer.writerow([key, sum([float(v) for v in val]) / len(val)])


def _evaluate_submission(predictor, iterator, output_stem, device):
    if type(device) is int:
        device = chainer.cuda.get_device(device)

    output_dir = os.path.abspath(os.path.dirname(output_stem))
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_fname = "{}.csv".format(output_stem)
    fileobj = open(output_fname, "w")
    csv_writer = csv.writer(fileobj)
    csv_writer.writerow(["image_name", "target"])

    iterator.reset()
    for batch in iterator:
        imgs, metas = concat_examples([b[:-1] for b in batch], device)
        with device:
            preds = predictor.predict(imgs, metas)
            for pred, data in zip(preds, batch):
                csv_writer.writerow([data[-1], pred[-1]])
    fileobj.close()
    return output_fname


def show_metrics(df,
                 class_labels=None,
                 true_header="true",
                 pred_header="pred",
                 roc_header=f"conf_{constants.Labels.malignant.name}"):
    """Show accuracy, confusion_matrix, precision, recall, f1, ROC
    Args :
        df (pd.DataFrame) : cnn result dataframe
        class_labels (list of string) :
        true_header (string) : dataframe's header name of true label
        pred_header (string) : dataframe's header name of predict label
        roc_header (string)  : dataframe's header name of calculating roc
    """
    accuracy = accuracy_score(df[true_header], df[pred_header])
    if class_labels is None:
        class_labels = sorted(set(df[true_header].unique() + df[pred_header].unique()))
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
    roc = roc_auc_score(df[true_header], df[roc_header])
    print(f"ROC-AUC score : {roc:.3f}")
