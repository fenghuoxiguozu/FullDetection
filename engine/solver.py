import torch.optim as optim
import math

class LrScheduler:
    def __init__(self,cfg,optimizer):
        self.lr = cfg["SOLVER"]["lr"]
        self.lrf = cfg["SOLVER"]["lrf"]
        self.epochs = cfg["epochs"]
        self.optimizer = optimizer
                       
    def get_lr(self):
        return lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1.0 - self.lrf) + self.lrf

    def get_scheduler(self):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.get_lr())


class SgdOptimizer:
    def __init__(self,cfg,model):
        self.lr = cfg["SOLVER"]["lr"]
        self.momentum = cfg["SOLVER"]["momentum"]
        self.weight_decay = cfg["SOLVER"]["weight_decay"]
        self.model =model
                      
    def get_pg(self):
        return [p for p in self.model.parameters() if p.requires_grad]
        
    def get_optimizer(self):
        return optim.SGD(self.get_pg(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay )



def build_scheduler(cfg,optimizer):
    return LrScheduler(cfg,optimizer).get_scheduler()
  
  
def build_optimizer(cfg,model):
    return SgdOptimizer(cfg,model).get_optimizer()