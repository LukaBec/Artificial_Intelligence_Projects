"""
CS311 Artificial Intelligence

By: Luka Becerra
"""
import argparse, math, os, re, string, zipfile
from typing import DefaultDict, Generator, Hashable, Iterable, List, Sequence, Tuple
from collections import defaultdict
import numpy as np
from sklearn import metrics


class Counts:
    def __init__(self):
        self.counts = defaultdict(int)
        self.num_docs = 0
        self.num_words = 0

    def probabilities(self, words, prob, pseudo):
        denom = self.num_words + len(self.counts)*pseudo
        for word in words:
            prob += np.log((self.counts.get(word, 0)+pseudo)/denom)
        return prob
        


class Sentiment:
    """Naive Bayes model for predicting text sentiment"""

    def __init__(self, labels: Iterable[Hashable]):
        """Create a new sentiment model

        Args:
            labels (Iterable[Hashable]): Iterable of potential labels in sorted order.
        """
        self.categories = {}
        for label in labels:
            self.categories[label] = Counts()
        pass

    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Normalize the string into a list of words.

        Args:
            example (str): Text input to split and normalize
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            List[str]: Normalized words
        """
        example = example.strip(string.punctuation).lower()
        return example.split()

    def add_example(self, example: str, label: Hashable, id:str = None):
        """Add a single training example with label to the model

        Args:
            example (str): Text input
            label (Hashable): Example label
            id (str, optional): File name from training/test data (may not be available). Defaults to None.
        """
        words = self.preprocess(example)
        category = self.categories[label]
        category.num_docs += 1
        for feature in words:
            category.counts[feature] += 1
            category.num_words += 1
        pass

    def predict(self, example: str, pseudo=0.0001, id:str = None) -> Sequence[float]:
        """Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels
        """
        #loop through each category and then loop through each word then calculate the prob, after it all normalize across each category
        #num word / len word + sudo
        words = self.preprocess(example)
        total_docs = sum(category.num_docs for category in self.categories.values())
        probs = []
        for category in self.categories.values():
            p = math.log(category.num_docs/total_docs)
            probs.append(category.probabilities(words,p,pseudo))
        return np.exp(np.subtract(probs,np.logaddexp.reduce(probs)))
                
        return [1.0, 0.0]

class CustomSentiment(Sentiment):
    def __init__(self, labels: Iterable[Hashable]):
        super().__init__(labels)



def process_zipfile(filename: str) -> Generator[Tuple[str, str, int], None, None]:
    """Create generator of labeled examples from a Zip file that yields a tuple with
    the id (filename of input), text snippet and label (0 or 1 for negative and positive respectively).

    You can use the generator as a loop sequence, e.g.

    for id, example, label in process_zipfile("test.zip"):
        # Do something with example and label

    Args:
        filename (str): Name of zip file to extract examples from

    Yields:
        Generator[Tuple[str, str, int], None, None]: Tuple of (id, example, label)
    """
    with zipfile.ZipFile(filename) as zip:
        for info in zip.infolist():
            # Iterate through all file entries in the zip file, picking out just those with specific ratings
            match = re.fullmatch(r"[^-]+-(\d)-\d+.txt", os.path.basename(info.filename))
            if not match or (match[1] != "1" and match[1] != "5"):
                # Ignore all but 1 or 5 ratings
                continue
            # Extract just the relevant file the Zip archive and yield a tuple
            with zip.open(info.filename) as file:
                yield (
                    match[0],
                    file.read().decode("utf-8", "ignore"),
                    1 if match[1] == "5" else 0,
                )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment analyzer")

    parser.add_argument(
        "--train",
        default="data/train.zip",
        help="Path to zip file or directory containing training files.",
    )
    parser.add_argument(
        "--test",
        default="data/test.zip",
        help="Path to zip file or directory containing testing files.",
    )
    parser.add_argument(
        "-m", "--model", default="base", help="Model to use: One of base or custom"
    )
    parser.add_argument("example", nargs="?", default=None)

    args = parser.parse_args()

    # Train model
    if args.model == "custom":
        model = CustomSentiment(labels=[0, 1])
    else:
        model = Sentiment(labels=[0, 1])
    for id, example, y_true in process_zipfile(
        os.path.join(os.path.dirname(__file__), args.train)
    ):
        model.add_example(example, y_true, id=id)

    # If interactive example provided, compute sentiment for that example
    if args.example:
        print(model.predict(args.example))
    else:
        predictions = []
        for id, example, y_true in process_zipfile(
            os.path.join(os.path.dirname(__file__), args.test)
        ):
            # Determine the most likely class from predicted probabilities
            predictions.append((id, y_true, np.argmax(model.predict(example,id=id))))

        # Compute and print accuracy metrics
        _, y_test, y_true = zip(*predictions)
        predict_metrics = compute_metrics(y_test, y_true)
        for met, val in predict_metrics.items():
            print(
                f"{met.capitalize()}: ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
