try:
    import colored_traceback.auto
except:
    pass
import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import nni
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from pytorch_memlab import MemReporter



#DEPRECATION WARNINGS
import warnings
warnings.filterwarnings("ignore")

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def import_module(name, config):
    return getattr(__import__("{}.{}".format(name, config[name]['module_name'])), config[name]['type'])

def mod_config(config, nni_params):
    if (nni_params == None):
        return config
    def recurse_dict(d, k, v):
        if (k in d):
            d[k] = v
            return d
        for kk, vv in d.items():
            if (type(vv) == collections.OrderedDict):
                d[kk] = recurse_dict(vv, k, v)
        return d
    for k, v in nni_params.items():

        if k in config:
            config[k] = v
            continue
        for kk, vv in config.items():
            if (type(vv) == collections.OrderedDict):
                config[kk] = recurse_dict(vv, k, v)
    return config

#def get_lr_lambda(model, config):
#    if config['lr_scheduler']['type'] != 'LambdaLR':
#        return None
#
#    if config['lr_scheduler']['args']['lr_lambda'] == 'fix':
#        lr_schedule = {0:0.1}
#        if (hasattr(model, 'lr_schedule')):
#            lr_schedule = model.lr_schedule

def main(config, nni_params={}):

    config._config = mod_config(config._config, nni_params)

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = import_module('model', config)(**config['model']['args'])#config.initialize('arch', module_arch)
    logger.info(model)
    #print(model)
    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    if config['lr_scheduler']['type'].lower() == 'custom':
        lr_scheduler = None # code lr_scheduler into trainer..
    else:
        lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    Trainer = import_module('trainer', config)

    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    params = {}

    try:
        params = nni.get_next_parameter()
    except:
        pass

    config = ConfigParser(args, options)

    config._config = mod_config(config._config, params)


    main(config)
