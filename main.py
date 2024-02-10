import argparse
import json

from models.model_builder import ModelBuilder
from common.trainer import Trainer
from common.metric import Metric
from common.data import get_dataloader
from config.config import PipelineConfig


def get_parser():
    arg = argparse.ArgumentParser()
    
    arg.add_argument(
        '--json_file', 
        dest='json_file', 
        required=True, 
        help="pipeline configuration json file"
    )
    return arg.parse_args()

if __name__ == '__main__':
    args = get_parser()
    pipeline_confg = PipelineConfig.from_json(args.json_file)
    train_dl, val_dl = get_dataloader(pipeline_confg)
    model_builder = ModelBuilder(pipeline_confg)
    model = model_builder.get()
    
    trainer = Trainer(
        model=model, 
        train_dataloader=train_dl, 
        eval_dataloader=val_dl, 
        loss_fn=model.get_loss_fn(), 
        epochs=pipeline_confg.epochs, 
        optimizer_clz=model.get_optim_clz(), 
        optim_params=model.get_optim_params_dict(),
        device=pipeline_confg.device, 
        metric_collection=Metric(pipeline_confg.device, task=pipeline_confg.task),
        verbose=pipeline_confg.verbose, 
        log_step=pipeline_confg.log_step,
        rating_threshold=pipeline_confg.rating_threshold
    )
    
    report = trainer.fit()
    json.dump(json.dumps(report), open(pipeline_confg.write_report_file, 'w'))