import logging
from pathlib import Path
import torch
import yaml
import quan
import util
import os
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import distribute_bn
from torch.nn.parallel import DistributedDataParallel as DDP
from model import create_model
from process import train, PerformanceScoreboard
import torch.nn.utils.prune as prune



def prune_network(model, train_loader,optimizer,lr_scheduler,criterion,args, method='iterative', prune_type='unstructured', 
                  num_iterations=5, target_reduction=0.5, initial_state_dict=None, prune_percentage=0.25, distill_criterion=None):
    monitors = None
    epoch = 0
    if method == 'netadapt':
        current_reduction = 0
        while current_reduction < target_reduction:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    if prune_type == 'unstructured':
                        prune.l1_unstructured(module, name='weight', amount=prune_percentage)
                    elif prune_type == 'structured':
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=prune_percentage, n=2, dim=0)

                    temp_reduction = 0.05
                    current_reduction += temp_reduction
                    if current_reduction >= target_reduction:
                        break

            # Fine-tuning step

                t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                                lr_scheduler, epoch, monitors, args, distill_criterion=distill_criterion)
                # Update learning rate and save checkpoint
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch+1, t_loss)
                    util.save_checkpoint(epoch, args.arch, model)

    elif method == 'rewinding':
        model.load_state_dict(initial_state_dict,strict = False)
        for epoch in range(num_iterations):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    if prune_type == 'unstructured':
                        prune.l1_unstructured(module, name='weight', amount=prune_percentage)
                    elif prune_type == 'structured':
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=prune_percentage, n=2, dim=0)

        # Rewind and fine-tune
            model.load_state_dict(initial_state_dict, strict=False)
            t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                            lr_scheduler, epoch, monitors, args, distill_criterion=distill_criterion)
            print(f"Iteration for pruning: {epoch}, Top 1 ACC: {t_top1}, Top 5 ACC: {t_top5}, Loss: {t_loss}")
                

    elif method == 'iterative':
        for epoch in range(num_iterations):
            # Pruning step
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    if prune_type == 'unstructured':
                        prune.l1_unstructured(module, name='weight', amount=prune_percentage)
                    elif prune_type == 'structured':
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=prune_percentage, n=2, dim=0)

        # Fine-tuning step after each pruning iteration
            t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                            lr_scheduler, epoch, monitors, args, distill_criterion=distill_criterion)
            print(f"Iteration for pruning: {epoch}, Top 1 ACC: {t_top1}, Top 5 ACC: {t_top5}, Loss: {t_loss}")
    

    return t_loss



def init_activation_scale_factors(model):
    pass


def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir /
                           'config_resnet50.yaml')
    
    monitors = None
    assert args.training_device == 'gpu', 'NOT SUPPORT CPU TRAINING NOW'

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

    args.device = 'cuda:0'

    args.world_size = 1
    args.rank = 0
    args.local_rank = 1

    assert args.rank >= 0, 'ERROR IN RANK'
    model = create_model(args)  # main model
    start_epoch = 0

    modules_to_replace = quan.find_modules_to_quantize(model, args)

    model.cuda()

    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, 'cuda', lean=args.resume.lean)

   
    if args.local_rank == 0:
        print(model)
    
    if args.freeze_weights:
        for name, para in model.named_parameters():
            if 'quan_' not in name and 'bn' not in name:
                para.requires_grad = False

    train_loader, val_loader, test_loader = util.data_loader.load_data(
        args.dataloader)

    # Define loss function (criterion) and optimizer
    criterion = LabelSmoothingCrossEntropy(args.smoothing).cuda()
    distill_criterion = SoftTargetCrossEntropy().cuda()

    optimizer = create_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

       
    initial_state_dict = model.state_dict()
    pruning_options = [
                {'method': 'iterative', 'prune_type': 'structured'},
                {'method': 'rewinding', 'prune_type': 'structured'},
                {'method': 'rewinding', 'prune_type': 'unstructured'},
                {'method': 'iterative', 'prune_type': 'unstructured'},
    ]
    for epoch in range(30):
        t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                                    lr_scheduler, epoch, monitors, args, distill_criterion=distill_criterion)
        print(f"Iteration for no pruning: {epoch}, Top 1 ACC: {t_top1}, Top 5 ACC: {t_top5}, Loss: {t_loss}")
                        # Update learning rate and save checkpoint
        if lr_scheduler is not None:
            lr_scheduler.step(epoch+1, t_loss)
            util.save_checkpoint(epoch, args.arch, model)
    for option in pruning_options:
        method = option['method']
        prune_type = option['prune_type']
        
        print(f"Executing {method} method with {prune_type} pruning")
        loss = prune_network(model,train_loader,optimizer,lr_scheduler,criterion,args,method=method,prune_type=prune_type
                            ,distill_criterion=distill_criterion,initial_state_dict=initial_state_dict)


if __name__ == "__main__":
    main()