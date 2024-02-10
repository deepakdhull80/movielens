import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self,
                model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                eval_dataloader: torch.utils.data.DataLoader = None,
                loss_fn = torch.nn.functional.mse_loss,
                epochs: int = 5,
                optimizer_clz = torch.optim.SGD,
                optim_params: dict = {'lr':1e-2},
                device: torch.device = torch.device("cpu"),
                metric_collection = None,
                verbose = 1,
                log_step = 500,
                *args, **kwargs
                ):
        self.log_step = log_step
        self.verbose = verbose
        self.model = model.to(device)
        self.train_dl = train_dataloader
        self.eval_dl = eval_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer_clz(self.model.parameters(), **optim_params)
        self.epochs = epochs
        self.device = device
        self.metric_collection = metric_collection.to(device) if metric_collection else None
        self.rating_threshold = kwargs.get('rating_threshold',3)

    def compute_loss(self, batch, output):
        loss = {}
        for k in output:
            if k in batch:
                loss[k] = self.loss_fn(output[k].view(-1).float(), batch[k].view(-1).float())
        return loss

    def compute_metric(self, batch, output, show_tensor=False):
        res = {}
        for k in output:
            if k in batch:
                y = (batch[k] >= self.rating_threshold).long()
                if show_tensor:
                    print(batch[k].sum(), batch[k].shape[0])
                res[k] = self.metric_collection(output[k].view(-1), y.view(-1))
        return res
    
    def set_output(self, _iter, loss, avg_loss, train=True):
        _iter.set_description(
            "{}: step loss: {:.3f}, avg_loss: {:.3f}".format(
                "Train" if train else "Eval",
                loss,
                avg_loss
            )
        )
    def train_epoch(self):
        self.model = self.model.train()
        
        _iter = tqdm(self.train_dl)
        n_batches = len(self.train_dl)
        total_loss = 0
        for _i, batch in enumerate(_iter):
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.compute_loss(batch, output)
            # multiple task we need better way to gives weights, ryt now it is uniform
            loss = sum([loss[l] for l in loss])/(len(loss) + 1e-6)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.cpu().item()
            
            self.set_output(_iter, loss.item(), total_loss/(_i+1), train=True)
            show_tensor = False
            if self.verbose > 0 \
                and _i % self.log_step == 0 and _i != 0:
                print(self.metric_collection.get())
                show_tensor = True
            with torch.no_grad():
                self.compute_metric(batch, output, show_tensor)
        
        final_metric = self.metric_collection.get()
        self.metric_collection.flush()
        
        return {
            'metric':final_metric,
            'loss': total_loss/(n_batches),
            'mode': 'Train'
        }
    
    @torch.no_grad()
    def eval_epoch(self):
        self.model = self.model.eval()
        
        _iter = tqdm(self.eval_dl)
        n_batches = len(self.eval_dl)
        total_loss = 0
        for _i, batch in enumerate(_iter):
            output = self.model(batch)
            loss = self.compute_loss(batch, output)
            # multiple task we need better way to gives weights, ryt now it is uniform
            loss = sum([loss[l] for l in loss])/(len(loss) + 1e-6)
            
            total_loss += loss.cpu().item()
            self.set_output(_iter, loss.item(), total_loss/(_i+1), train=False)
            
            show_tensor = False
            if self.verbose > 0 \
                and _i % self.log_step == 0 and _i != 0:
                print(self.metric_collection.get())
                show_tensor = True
            self.compute_metric(batch, output, show_tensor=show_tensor)
        
        final_metric = self.metric_collection.get()
        self.metric_collection.flush()
                
        return {
            'metric': final_metric,
            'loss': total_loss/(n_batches),
            'mode': 'Eval'
        }

    def fit(self) -> dict:
        report = {
            'train': [],
            'eval': []
        }
        for epoch in range(1, self.epochs+1):
            print("EPOCH:", epoch)
            train_report = self.train_epoch()
            report['train'].append(train_report)
            eval_report = self.eval_epoch()
            report['eval'].append(eval_report)
            print(train_report)
            print(eval_report)
        return report


if __name__ == '__main__':
    pass
    
    # trainer = Trainer(
    #     model=model, train_dataloader=traindl, eval_dataloader=testdl, 
    #     loss_fn=F.binary_cross_entropy_with_logits,
    #     metric_collection = Metric(),
    #     log_step = 100
    # )