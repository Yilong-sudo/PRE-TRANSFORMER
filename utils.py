import os
import torch


class BestMeter(object):
    """Computes and stores the best value"""
    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)


def load_checkpoint(model_path):
    return torch.load(model_path)


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def load_model_dict(model, ckpt):
    """安全地加载预训练模型，只加载兼容的参数"""
    checkpoint = torch.load(ckpt)
    model_dict = model.state_dict()
    
    # 过滤出存在于当前模型中的参数
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    # 更新当前模型的参数字典
    model_dict.update(pretrained_dict)
    
    # 加载更新后的参数
    model.load_state_dict(model_dict)
    
    print(f"Loaded {len(pretrained_dict)} parameters from pretrained model")
    missing_keys = set(checkpoint.keys()) - set(pretrained_dict.keys())
    if missing_keys:
        print(f"Skipped {len(missing_keys)} parameters not compatible with current model")


def load_model_dict_strict(model, ckpt):
    """严格加载模型（原始版本）"""
    model.load_state_dict(torch.load(ckpt))


def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x