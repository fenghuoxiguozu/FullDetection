from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import utils.comm as comm

def train_one_epoch(model,data_loader,device,epoch,optimizer,scheduler):
    model.train()
    sum_loss = torch.zeros(1).to(device)  # 累计损失
    loss_function = nn.CrossEntropyLoss()
    if comm.is_main_process():
        data_loader = tqdm(data_loader)
    
    for step,data in enumerate(data_loader):
        optimizer.zero_grad()
        images, labels = data
        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss = reduce_value(loss, average=True)
        sum_loss += loss.detach()
        
        if comm.is_main_process():
            data_loader.desc = "train epoch[{}] loss:{:.3f}".format(epoch + 1, sum_loss.item()/(step+1))
        
 
def val_one_epoch(model,data_loader,device,epoch):
    model.eval()
    acc_num = torch.zeros(1).to(device)
    nums = len(data_loader.dataset)
    with torch.no_grad():
        if comm.is_main_process():
            data_loader = tqdm(data_loader)
        for val_data in data_loader:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_y,val_labels.to(device)).sum()
        acc = reduce_value(acc_num, average=False)
        val_accurate = acc.item() / nums
        if comm.is_main_process():
            print('[epoch {}] val_accuracy: {:.3f}'.format(epoch+1,val_accurate))
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch+1))


def reduce_value(value, average=True):
    world_size = comm.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value
