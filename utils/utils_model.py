from ast import literal_eval
import torch
import os

def load_model(args, logging, models, results):
    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}
    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            if args.start_over:
                args.start_epoch = 0
            else:
                args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            sd = model.state_dict()

            ptd = {k:v for k, v in checkpoint['state_dict'].items() if k in sd}
            if (len(ptd) == 0):
                ptd = {k.strip('module').strip('.'):v for k, v in checkpoint['state_dict'].items() if k.strip('module').strip('.')}

            for k, v in sd.items():
                if (k not in ptd):
                    ptd[k] = v

            #import pdb; pdb.set_trace()
            model.load_state_dict(ptd)
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, args.start_epoch)
        else:
            logging.error("no checkpoint found at '%s'", args.resume)
    model.type(args.type)
    return model, model_config

def filt_alpha_weight(model):
    alpha = []
    weights = []
    layers = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ('alpha' in name):

                n_split = name.split('alpha')
                nn = n_split[0]
                if (nn not in layers.keys()):
                    layers[nn] = []
                alpha.append(param)

    for name, param in model.named_parameters():
        if param.requires_grad:
            n_split = name.split('weight')
            nn = n_split[0]
            if (nn in list(layers.keys())):
                weights.append(param)

    return alpha, weights


class Freezer():
    def __init__(self, model):
        self.model = model
        self.requires_grad_dict_w = {}
        self.alpha = {}

        for name, param in model.named_parameters():
            if (param.requires_grad and 'alpha' not in name):
                self.requires_grad_dict_w[name] = param
            elif ('alpha' in name):
                self.alpha[name] = param
    def freeze_weights(self):
        for k, v in self.requires_grad_dict_w.items():
            v.requires_grad = False

    def unfreeze_weights(self):
        for k, v in self.requires_grad_dict_w.items():
            v.requires_grad = True

    def freeze_scales(self):
        for k, v in self.alpha.items():
            v.requires_grad = False

    def unfreeze_scales(self):
        for k, v in self.alpha.items():
            v.requires_grad = True

def get_reg(alpha, weights, reg_type):
    reg = 0
    if reg_type == 'l1':
        for a, w in zip(alpha, weights):
            reg += torch.sum(torch.abs(torch.abs(w) - a))
    elif reg_type == 'l2':
        for a, w in zip(alpha, weights):
            reg += torch.sum(torch.abs(torch.mul(w, w) - a))
    elif reg_type == 'l2_2':
        for a, w in zip(alpha, weights):
            reg += torch.sum(torch.mul(w,w) - a)
    else:
        reg = 0
    return reg

def get_reg_lr(epoch, config):
    if (config  == ''):
        return  0
    if (config['type'] == 'linear'):
        if (config['epoch_start'] < epoch):
            return (epoch + 1 - config['epoch_start']) * config['init_lr']
    elif (config['type'] == 'stepLR'):
        lr_schedule = config['lr_schedule']
        lr = 0
        for k, lr in lr_schedule.items():
            if int(epoch) < int(k):
                return lr
    return 0

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}
def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


