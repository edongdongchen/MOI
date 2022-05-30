from datetime import datetime

# --------------------------------
# logger
# --------------------------------
def get_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H:%M:%S')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}={avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_epochs, meters, prefix=""):
        self.epoch_fmtstr = self._get_epoch_fmtstr(num_epochs)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch):
        entries = [get_timestamp()]
        entries += [self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        entries += [self.prefix]
        print('\t'.join(entries))

    def _get_epoch_fmtstr(self, num_epochs):
        num_digits = len(str(num_epochs // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_epochs) + ']'

