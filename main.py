from subprocess import call

from parameter import *
from trainer import Trainer
from data_loader import dataloader
# from utils import *

def main(config):

    ## TODO: This is temporary solution
    if config.exec_data_setter:
        args = config.dataset.split()
        call(['bash', 'data_setter.sh', args[0], args[1]]) ## TODO
        data_loader = dataloader(config.dataset, config.img_rootpath,
                                 config.img_size, config.batch_size)
         ## TODO
    else:
        data_loader = dataloader(config.dataset, config.img_rootpath,
                                 config.img_size, config.batch_size)

    if config.train:
        if config.model == 'can':
            trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        ## TODO: short tester
        pass


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
