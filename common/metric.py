import torch
import torchmetrics
from torchmetrics.classification import average_precision, ROC, Recall, Accuracy

class Metric:
    def __init__(self, device=torch.device("cpu"), task="binary", type="regression"):
        self.device = device
        self.task = task
        self.type = type
        self.counter = 0
        if self.type == 'regression':
            self._metric = {
                "mse": torchmetrics.functional.mean_squared_error,
                "r2_score": torchmetrics.functional.r2_score
                
            }
            self._accelerator = {
                "mse": 0,
                "r2_score": 0
            }
        elif self.type == 'classification':
            self._metric = {
                "map": average_precision.AveragePrecision(task=task),
                "accuracy": Accuracy(task=task),
                "recall": Recall(task=task),
                "auc": torchmetrics.functional.auc
            }
            self._accelerator = {
                "map": 0,
                "accuracy": 0,
                "recall": 0,
                "auc": 0
            }
        
    def __call__(self, output, actual) -> dict:
        out = {}
        for metric_name, mod in self._metric.items():
            out[metric_name] = mod(output, actual)
            self._accelerator[metric_name] += out[metric_name]
        self.counter += 1
        return out
    
    def get(self):
        return {
            k:v/self.counter for k,v in self._accelerator.items()
        }

    def to(self, device):
        for i in self._metric:
            if hasattr(self._metric[i], "to"):
                self._metric[i].to(device)
        return self
    
    def flush(self):
        return Metric(
            device=self.device,
            task=self.task,
            type=self.type
        )


if __name__ == "__main__":
    metric = Metric()
    output, actual = torch.randn(20, 1).view(-1), torch.randint(0, 2, size=(20, 1)).view(-1)
    print(actual, output)
    print(metric(output, actual))
    print(metric.get())