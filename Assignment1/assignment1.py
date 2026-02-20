# -*- coding: utf-8 -*-

#basic libraries to help with the process
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

#setting the option to be able to read the full text
pd.set_option("display.max_colwidth", None)

#setting up a random seed
SEED = 36


#loading the datasets
print("Loading datasets...")
train = load_dataset("sh0416/ag_news", split="train")
test  = load_dataset("sh0416/ag_news", split="test")

#making them pandas datasets (easier to work with)
train_start = pd.DataFrame(train)
test = pd.DataFrame(test)

"""## Preprocessing"""

#checking if any nulls in dataset
print("\n")
print("Checking for nulls in the train dataset...")
print("\n")
print(train_start.info())
print("\n")
print("Checking for nulls in the test dataset...")
print("\n")
print(test.info())

#first look at the dataset
#notice how there seems to be some "\" in between words in some of the descriptions, we'll have to fix that
print("\n")
print("First look at the train dataset:")
print("\n")
print(train_start)
print("\n")
print("Notice how there seems to be some \"\\\" in between words in some of the descriptions, we'll have to fix that")

#combining title and description (more info)
title = train_start["title"].astype(str)
desc = train_start["description"].astype(str)

#removing the "\"
title = title.str.replace("\\", " ", regex=False)
desc = desc.str.replace("\\", " ", regex=False)

#we can add weights later if necessary
train_start["x_text"] = title + " " + desc

#lower case (import for word count)
train_start["x_text"] = train_start["x_text"].str.lower()

#doing the same for test split

title = test["title"].astype(str)
desc = test["description"].astype(str)

title = title.str.replace("\\", " ", regex=False)
desc = desc.str.replace("\\", " ", regex=False)

test["x_text"] = title + " " + desc

test["x_text"] = test["x_text"].str.lower()

#the combined text
print("\n")
print("Combined text (title + description) for the train dataset:")
print("\n")
print(train_start["x_text"].head())

#splits the train set into train and validation on a 90/10 split, this split is with seed 36
train, val = train_test_split(train_start, test_size=0.1, random_state = SEED)

"""## Metrics"""

#setting up accuracy function
def accuracy(y_true, y_pred):
    #set y_true and y_pred as numpy arrays so we can use (i, len(y_pred)) as positional arguments
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    #initialize correct prediction count as 0
    same = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            #if prediction was right, increse the count
            same += 1
    #return the accuracy
    return same / len(y_pred)

#setting a function that returns True Positives, True Negatives, False Positives and False Negatives, per label
def metrics(label, y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == label and yp == label:
            tp += 1
        elif yt != label and yp == label:
            fp += 1
        elif yt == label and yp != label:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn

#calculates the f1-score given a label
def f1(label, y_true, y_pred):
    tp, fp, fn, tn = metrics(label, y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    return 2 * (precision * recall) / (precision + recall)


#calculates the macro f1 (average f1)
def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    sum = 0
    for label in labels:
        sum += f1(label, y_true, y_pred)
    return sum / 4

# uses sklearn's confusion matrix function
def conf_matrix(y_true, y_pred, class_names=["World", "Sports", "Business", "Sci/Tech"], normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        ax=ax,
        colorbar=True
    )

    #ax.set_title("Normalized Confusion Matrix (Test Set)" if normalize else "Confusion Matrix (Test Set)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

"""## TF-IDF

Feature extraction
"""

print("\n")
print("Creating feature vectors with TF-IDF...")
# Setting up TF-IDF with 10k features and removing common stop words
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Transforming the data - only fitting on train to avoid leaking info
X_train = vectorizer.fit_transform(train['x_text'])
X_dev = vectorizer.transform(val['x_text'])
X_test = vectorizer.transform(test['x_text'])

# Target labels
y_train = train['label']
y_dev = val['label']
y_test = test['label']

print(f"Vectors created with {X_train.shape[1]} features.")

"""Model training"""

#sets up a grid search for parameter c
c_grid = {"C": [0.1, 1, 10]}
# Baseline 1: Logistic Regression
# Added more iterations so it doesn't complain about convergence
lr_clf = LogisticRegression(max_iter=1000)
lr_clf = GridSearchCV(lr_clf, c_grid, cv=5, scoring="accuracy")
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_dev)

# Baseline 2: Linear SVM
# Usually more solid for text classification
svm_clf = LinearSVC(random_state=SEED)
svm_clf = GridSearchCV(svm_clf, c_grid, cv=5, scoring="accuracy")
svm_clf.fit(X_train, y_train)
svm_preds = svm_clf.predict(X_dev)

print("\n")
print("Models trained.")
print("\n")
print(f"LR Acc: {accuracy(y_dev, lr_preds):.4f}")
print(f"LR Macro F1: {macro_f1(y_dev, lr_preds):.4f}")
print(f"LR best value for C: {lr_clf.best_params_}")
print("--------------------------------")
print(f"SVM Acc: {accuracy(y_dev, svm_preds):.4f}")
print(f"SVM Macro F1: {macro_f1(y_dev, svm_preds):.4f}")
print(f"SVM best value for C: {svm_clf.best_params_}")

"""Final Results & Error Analysis"""

# Final check on the test set using the SVM
test_predictions = svm_clf.predict(X_test)
print("\n")
print("--- Final Scores ---")
print(f"Test Acc using SVM: {accuracy(y_test, test_predictions):.4f}")
print(f"Test Macro-F1 using SVM: {macro_f1(y_test, test_predictions):.4f}")

# Confusion matrix to see the overlap between classes
print(conf_matrix(y_test, test_predictions))

# Error analysis: pulling 20 random-ish mistakes to check manually
# Just mapping numbers to names so I can actually read it
class_names = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
wrong_idx = [i for i, (p, a) in enumerate(zip(test_predictions, y_test)) if p != a]

print("\n--- Examples of mistakes ---")

def show_confusion_examples(val_df, y_true, y_pred, class_names, pairs, k=10, seed=0, text_col="x_text", max_chars=180):
    rng = np.random.default_rng(seed)

    # make them plain arrays for safe indexing
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for pred_label, true_label in pairs:
        idx = np.where((y_pred == pred_label) & (y_true == true_label))[0]

        print(f"\n=== predicted={class_names[pred_label]} | true={class_names[true_label]} | count={len(idx)} ===")
        print("\n")
        if len(idx) == 0:
            continue

        take = idx if len(idx) <= k else rng.choice(idx, size=k, replace=False)

        for j, ix in enumerate(take, 1):
            print(f"#{j}: true={class_names[true_label]} | pred={class_names[pred_label]}")
            print(val_df.iloc[ix][text_col][:max_chars] + "...")
            print("-" * 15)

# example pairs: 10 examples for each confusion type you pick
pairs = [
    (2, 1),  # predicted Sports, true World
    (2, 3),
    (2, 4),
    (1, 3),
    (1, 2), # predicted World, true Sports
    (1, 4),
    (3, 1),
    (3, 2),
    (3, 4),  # predicted Business, true Sci/Tech
    (4, 1),
    (4, 2),
    (4, 3),  # predicted Sci/Tech, true Business
]

show_confusion_examples(val, y_dev, svm_preds, class_names, pairs, k=10, seed=SEED)
