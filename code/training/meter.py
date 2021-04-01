import numpy as np
from utils.metrics import dice_score


class SegmentationMeter:
    """
    Meter to handle predictions & metrics.
    """
    def __init__(self, threshold=0.5):
        """
        Constructor

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """
        self.threshold = threshold
        self.reset()

    def update(self, y_batch, preds):
        """
        Update ground truths and predictions

        Args:
            y_batch (tensor): Truths.
            preds (tensor): Predictions.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        self.y_mask.append(y_batch.cpu().numpy())
        self.pred_mask.append(preds.cpu().numpy())

    def concat(self):
        """
        Concatenates everything.
        """
        self.pred_mask = np.concatenate(self.pred_mask)
        self.y_mask = np.concatenate(self.y_mask)

    def compute(self):
        """
        Computes the metrics.

        Returns:
            dict: Metrics dictionary.
        """
        self.concat()
        self.metrics["dice"] = [dice_score(self.pred_mask, self.y_mask)]
        return self.metrics

    def reset(self):
        """
        Resets everything.
        """
        self.pred_mask = []
        self.y_mask = []
        self.metrics = {
            "dice": [0],
        }
