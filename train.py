import yaml
import torch
import torch.multiprocessing as mp
import utils.comm as comm
from backbone.build import build_backbone
from data.dataset import create_dataloader
from engine.train_util import train_one_epoch,val_one_epoch
from engine.solver import build_optimizer,build_scheduler


def main(rank,cfg):
    device = torch.device('cuda',rank)
    train_loader = create_dataloader(cfg, "train")
    val_loader = create_dataloader(cfg, "val")
    
    model = build_backbone(cfg)
    model = comm.init_env(rank,cfg,model)
    
    optimizer = build_optimizer(cfg,model)
    scheduler = build_scheduler(cfg,optimizer)
    
    epochs = cfg["epochs"]
    for epoch in range(epochs):
        # train
        train_one_epoch(model,train_loader,device,epoch,optimizer,scheduler)
        # validate
        val_one_epoch(model,val_loader,device,epoch)


if __name__ == "__main__":
    cfg = yaml.safe_load(open('imagenet.yaml'))

    mp.spawn(main, nprocs=cfg["world_size"], args=(cfg,))

