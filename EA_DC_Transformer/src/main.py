"""
Written by Zhuo Li for Evoving Attention Dilated Convolution Transformer

reference to the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentError
import os

from PIL.Image import FASTOCTREE
import pdb
import logging
from numpy import dtype
import os
import sys
import time
import numpy as np
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils,utils_ea
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ea_dc_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")


def main(config, pass_index):

    global train_indices_list, val_indices_list, raw_my_data, raw_test_data, raw_val_data, test_indices, nsplits
    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout

    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    global isDataProcessed
    if isDataProcessed == False:
        isDataProcessed = True
        logger.info("Loading and preprocessing data ...")
        config['pattern'] = 'TRAIN'
        config['val_pattern'] = 'TRAIN'
        config['test_pattern'] = 'TEST'

        data_class = data_factory[config['data_class']]
        raw_my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
        feat_dim = raw_my_data.feature_df.shape[1]  # dimensionality of data features

        #when for cluster labels_df = None so task ! = classification
        if config['task'] == 'classification':
            validation_method = 'StratifiedShuffleSplit'
            labels = raw_my_data.labels_df.values.flatten()
        else:
            validation_method = 'ShuffleSplit'
            labels = None

        raw_val_data = raw_my_data
        val_indices = []
        if config['test_pattern']:  # used if test data come from different files / file patterns
            raw_test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
            test_indices = raw_test_data.all_IDs
        if config['test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
            test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
            try:
                test_indices = [int(ind) for ind in test_indices]  # integer indices
            except ValueError:
                pass  # in case indices are non-integers
            logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
        if config['val_pattern']:  # used if val data come from different files / file patterns
            # val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
            val_indices = raw_val_data.all_IDs

        config['val_ratio'] = 0.3
        # random split data three times for robustness of results
        nsplits = 3
        train_indices_list, val_indices_list, test_indices = split_dataset(data_indices=raw_my_data.all_IDs,
                                                                    validation_method=validation_method,
                                                                    n_splits=nsplits,
                                                                    validation_ratio=config['val_ratio'],
                                                                    test_set_ratio=config['test_ratio'],  # used only if test_indices not explicitly specified
                                                                    test_indices=test_indices,
                                                                    random_seed=1337,
                                                                    labels=labels)
   


    # else:
    #     print("no need to process data")

    logger.info("{} samples may be used for training".format(len(train_indices_list[0])))
    logger.info("{} samples will be used for validation".format(len(val_indices_list[0])))
    logger.info("{} samples will be used for testing".format(len(test_indices)))
    

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return
    

    start_epoch = 0

    val_best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    val_metrics = []  # (for validation and test) list of lists: for each epoch, stores metrics like loss, ...
    val_best_metrics = {}
    test_best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    val_tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    for k_fold in range(nsplits):
        
        logger.info("Creating model ...")
        model = model_factory(config, raw_my_data)

        if config['freeze']:
            for name, param in model.named_parameters():
                if name.startswith('output_layer'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        logger.info("Model:\n{}".format(model))
        logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
        logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
        

        # Initialize optimizer

        if config['global_reg']:
            weight_decay = config['l2_reg']
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config['l2_reg']

        optim_class = get_optimizer(config['optimizer'])
        optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

     
        lr_step = 0  # current step index of `lr_step`
        lr = config['lr']  # current learning step
        # Load model and optimizer state
        
        model.to(device)

        loss_module = get_loss_module(config)

        #build dataset
        train_indices = train_indices_list[k_fold]
        val_indices = val_indices_list[k_fold]


        normalizer = None
        if config['norm_from']:
            with open(config['norm_from'], 'rb') as f:
                norm_dict = pickle.load(f)
            normalizer = Normalizer(**norm_dict)
        elif config['normalization'] is not None:
            normalizer = Normalizer(config['normalization'])
            my_data = raw_my_data
            my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
            if not config['normalization'].startswith('per_sample'):
                # get normalizing values from training set and store for future use
                norm_dict = normalizer.__dict__
                # with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                #     pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
        if normalizer is not None:
            if len(val_indices):
                val_data = raw_val_data
                val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
            if len(test_indices):
                test_data = raw_test_data
                test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        val_dataset = dataset_class(val_data, val_indices)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=config['num_workers'],
                                pin_memory=True,
                                collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

        train_dataset = dataset_class(my_data, train_indices)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=config['batch_size'],
                                shuffle=True,
                                num_workers=config['num_workers'],
                                pin_memory=True,
                                collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    num_workers=config['num_workers'],
                                    pin_memory=True,
                                    collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

        trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                    print_interval=config['print_interval'], console=config['console'])

        val_evaluator = runner_class(model, val_loader, device, loss_module,
                                        print_interval=config['print_interval'], console=config['console'])
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])

        

        # Evaluate on validation before training
        if start_epoch == 0:
            val_aggr_metrics, val_best_metrics, val_best_value = validate(val_evaluator, val_tensorboard_writer, config, val_best_metrics,
                                                                val_best_value, epoch=0)
            val_metrics_names, val_metrics_values = zip(*val_aggr_metrics.items())
           
            with torch.no_grad():
                test_aggr_metrics = test_evaluator.evaluate(epoch_num=0, keep_all=False)
            # print_str = 'Test Summary: '
            print_str = 'Epoch {} Test Summary: '.format(0)
            
            cur_repr = test_aggr_metrics['reprs']
            cur_targets = test_aggr_metrics['targets'] 
            del test_aggr_metrics['reprs']
            del test_aggr_metrics['targets'] 
            for k, v in test_aggr_metrics.items():
                val_metrics_values += (v,)
                print_str += '{}: {:8f} | '.format(k, v)

            if(val_best_value == val_aggr_metrics[config['key_metric']]):
                test_best_value = test_aggr_metrics[config['key_metric']]
                np.save(os.path.join(config["output_dir"], 'reprs.npy'), cur_repr)
                np.save(os.path.join(config["output_dir"], 'targets.npy'), cur_targets)

            val_metrics.append(val_metrics_values) 
            logger.info(print_str)
        
        logger.info('Starting training...')
        for epoch in tqdm(range(start_epoch + 1, start_epoch + config["epochs"] + 1), desc='Training Epoch', leave=False):
            mark = epoch if config['save_all'] else 'last'
            epoch_start_time = time.time()
            aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
            epoch_runtime = time.time() - epoch_start_time
            
            print_str = 'Epoch {} Training Summary: '.format(epoch)
            for k, v in aggr_metrics_train.items():
                val_tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
                print_str += '{}: {:8f} | '.format(k, v)
            logger.info(print_str)
            logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))

            total_epoch_time += epoch_runtime
            avg_epoch_time = total_epoch_time / (epoch - start_epoch)
            avg_batch_time = avg_epoch_time / len(train_loader)
            avg_sample_time = avg_epoch_time / len(train_dataset)

            print("Model: {}, Data: {}, D_model: {}".format(config["model"],config["name"],config["d_model"]))

            logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
            logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
            logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

            # evaluate if first or last epoch or at specified interval
            if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
                val_aggr_metrics, val_best_metrics, val_best_value = validate(val_evaluator, val_tensorboard_writer, config,
                                                                    val_best_metrics, val_best_value, epoch)
                val_metrics_names, val_metrics_values = zip(*val_aggr_metrics.items())

                with torch.no_grad():
                    test_aggr_metrics = test_evaluator.evaluate(epoch_num=epoch, keep_all=False)
   
                print_str = 'Epoch {} Test Summary: '.format(epoch)
                # print_str = 'Test Summary: '
                cur_repr = test_aggr_metrics['reprs']
                cur_targets = test_aggr_metrics['targets'] 
                del test_aggr_metrics['reprs']
                del test_aggr_metrics['targets'] 
                for k, v in test_aggr_metrics.items():
               
                    val_metrics_values += (v,)
                    print_str += '{}: {:8f} | '.format(k, v)
                val_metrics.append(val_metrics_values) 

                if(val_best_value == val_aggr_metrics[config['key_metric']]):
                    test_best_value = test_aggr_metrics[config['key_metric']]
                    np.save(os.path.join(config["output_dir"], 'reprs.npy'), cur_repr)
                    np.save(os.path.join(config["output_dir"], 'targets.npy'), cur_targets)
         
            # Learning rate scheduling
            if epoch == config['lr_step'][lr_step]:
                # utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
                lr = lr * config['lr_factor'][lr_step]
                if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                    lr_step += 1
                logger.info('Learning rate updated to: ', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Difficulty scheduling
            if config['harden'] and check_progress(epoch):
                train_loader.dataset.update()
                val_loader.dataset.update()
        start_epoch = 0
 
        # Export evolution of metrics over epochs
        if k_fold == (nsplits-1):
            list_metrics_names = []
            for element in val_metrics_names:
                list_metrics_names.append("val_" + element)
            for k, v in test_aggr_metrics.items():
                list_metrics_names.append("test_" + k) 
            header = list_metrics_names
            metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] +"_" + str(pass_index) +  ".xls")
            book = utils.export_performance_metrics(metrics_filepath, val_metrics, header, sheet_name="metrics")

    
    
    global total_best_epoch, total_best_test_metric, passes, total_best_val_metric, run_times
    total_best_epoch += val_best_metrics['epoch']
    total_best_val_metric += val_best_metrics[config['key_metric']]
    total_best_test_metric  += test_best_value
    passes += 1

    if passes == run_times:
   
        total_best_epoch = total_best_epoch // run_times
        total_best_val_metric = total_best_val_metric/run_times
        total_best_test_metric = total_best_test_metric/run_times
        # Export record metrics to a file accumulating records from all experiments
        utils_ea.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                        config["model"], config["batch_size"], config["lr"], config["d_model"], config["num_layers"],config["kernel_size"], config["hidden_size"], config["isdilated"],
                        config["alpha"], config["beta"], config["k"],
                        total_best_epoch,total_best_val_metric, total_best_test_metric, comment=config['comment'])
        
    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], total_best_test_metric, total_best_val_metric))
    logger.info('All Done!')
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return total_best_test_metric



if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object
   
    #initialize
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    
    run_times = 3

    args.data_class = 'tsra'
    args.lr = 0.001
    args.optimizer = 'RAdam'
    args.pos_encoding = 'learnable'

    if(args.task == 'classification'):
        args.key_metric = "accuracy"
    if(args.task == 'regression'):
        args.key_metric  = "loss"

    isDataProcessed = False

    bare_output_dir = args.task + "/experiments"
    os.makedirs(bare_output_dir, exist_ok=True)
    args.comment = args.task 

    args.data_dir = "src/datasets/" + args.task + "/" + args.data
    if not os.path.exists(args.data_dir):
        raise ValueError("Invalid data directory!")

    args.records_file = args.task + "_"  + args.data + "_0321_records.xls"

    bare_data_dir = args.data_dir

        
    isDataProcessed = False
    args.experiment_name = bare_data_dir.split('/')[-1]
    args.data_dir  = bare_data_dir
    args.name = args.experiment_name + "_" + args.task

    # search for d_model parameter on valid metric
    for d_model_type in [128,64]:
        args.d_model = d_model_type
        args.hidden_size = d_model_type
        total_best_epoch = 0
        total_best_test_metric = 0
        total_best_val_metric = 0
        passes = 0
        for i  in tqdm(range(run_times)):
            args.output_dir = bare_output_dir
            config = setup(args)
           
            main(config,i)
    