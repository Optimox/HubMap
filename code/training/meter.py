from utils.metrics import dice_score_tensor


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
        Updates the metric.

        Args:
            y_batch (tensor): Truths.
            preds (tensor): Predictions.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        self.dice += dice_score_tensor(preds, y_batch, threshold=self.threshold) * preds.size(0)
        self.count += preds.size(0)

    def compute(self):
        """
        Computes the metrics.

        Returns:
            dict: Metrics dictionary.
        """
        self.metrics["dice"] = [self.dice / self.count]
        return self.metrics

    def reset(self):
        """
        Resets everything.
        """
        self.dice = 0
        self.count = 0
        self.metrics = {
            "dice": [0],
        }
        return self.metrics
