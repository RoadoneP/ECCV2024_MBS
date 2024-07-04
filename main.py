import os
import utils
import random
import logging
import argparse

import datetime
import time
import math

import numpy as np
from omegaconf import OmegaConf

from metrics import StreamSegMetrics

import torch
from torch.utils import data
import torch.nn.functional as F

from utils import ext_transforms as et
from utils.tasks import get_tasks

from datasets import VOCSegmentation
from datasets import ADESegmentation

from core import Segmenter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from utils import imutils
from utils.utils import AverageMeter

# argment parser
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='./configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--log', default='test.log')
parser.add_argument('--backend', default='nccl')

# calculate eta
def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

# logger function
def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

# train/val/test data prepare
def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.dataset.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.dataset.crop_size, opts.dataset.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.train.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.dataset.crop_size),
            et.ExtCenterCrop(opts.dataset.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    if opts.dataset.name == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset.name == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    
    dataset_dict['val'] = dataset(opts=opts, image_set='val', transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    return dataset_dict

# validate function
def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            outputs, _, _, _ = model(images)
            
            if opts.train.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
        
    return score

# train function
def train(opts):
    writer = SummaryWriter('runs/'+ str(args.log))
    num_workers = 4 * len(opts.gpu_ids)
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    # Get the target classes for the current task and step
    target_cls = get_tasks(opts.dataset.name, opts.task, opts.curr_step)
    
    # Calculate the number of classes for each step
    opts.num_classes = [len(get_tasks(opts.dataset.name, opts.task, step)) for step in range(opts.curr_step+1)]
    opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:]
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset.name, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset.name, opts.task, step)) for step in range(opts.curr_step+1))
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bg_label = 0
    
    if args.local_rank==0:
        print("==============================================")
        print(f"  task : {opts.task}")
        print(f"  step : {opts.curr_step}")
        print("  Device: %s" % device)
        print( "  opts : ")
        print(opts)
        print("==============================================")

    # Initialize the model with the specified backbone and number of classes
    model = Segmenter(backbone=opts.train.backbone, num_classes=opts.num_classes,
                pretrained=True)
    
    if opts.curr_step > 0:
        """ load previous model """
        model_prev = Segmenter(backbone=opts.train.backbone, num_classes=list(opts.num_classes)[:-1],
                pretrained=True)
    else:
        model_prev = None
    
    get_param = model.get_param_groups()
    
    if opts.curr_step > 0:
        param_group = [{"params": get_param[0], "lr": opts.optimizer.learning_rate*opts.optimizer.inc_lr}, # Encoder
                    {"params": get_param[1], "lr": opts.optimizer.learning_rate*opts.optimizer.inc_lr}, # Norm
                    {"params": get_param[2], "lr": opts.optimizer.learning_rate*opts.optimizer.inc_lr}] # Decoder
    else:
        param_group = [{"params": get_param[0], "lr": opts.optimizer.learning_rate}, # Encoder
                    {"params": get_param[1], "lr": opts.optimizer.learning_rate}, # Norm
                    {"params": get_param[2], "lr": opts.optimizer.learning_rate}] # Decoder
    
    # Initialize the optimizer with the parameter groups
    optimizer = torch.optim.SGD(params=param_group, 
                            lr=opts.optimizer.learning_rate,
                            weight_decay=opts.optimizer.weight_decay, 
                            momentum=0.9, 
                            nesterov=True)
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        
        if args.local_rank==0:
            print("Model saved as %s" % path)

    utils.mkdir('checkpoints')    
    # Restore
    best_score = -1
    cur_epochs = 0
    
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    # model load from checkpoint if opts_curr_step == 0 
    if opts.curr_step==0 and (opts.ckpt is not None and os.path.isfile(opts.ckpt)):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(checkpoint, strict=True)
        
        if args.local_rank==0:
                print("Curr_step is zero. Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    
    # model load from checkpoint if opts_curr_step > 0
    if opts.curr_step > 0:
        opts.ckpt = ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step-1)
    
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
            model_prev.load_state_dict(checkpoint, strict=True)         
            
            # Transfer the background class token if weight transfer is enabled
            if opts.train.weight_transfer:
                curr_head_num = len(model.decoder.cls_emb) - 1
                class_token_param = model.state_dict()[f"decoder.cls_emb.{curr_head_num}"]
                for i in range(opts.num_classes[-1]):
                    class_token_param[:, i] = checkpoint["decoder.cls_emb.0"]
                        
                checkpoint[f"decoder.cls_emb.{curr_head_num}"] = class_token_param
                    
            model.load_state_dict(checkpoint, strict=False)
                
            if args.local_rank==0:
                print("Model restored from %s" % opts.ckpt)
            del checkpoint  # free memory
        else:
            if args.local_rank==0:
                print("[!] Retrain")
    
    if opts.curr_step > 0:
        model_prev.to(device)
        model_prev.eval()
    
        for param in model_prev.parameters():
            param.requires_grad = False
                
    if args.local_rank==0 and opts.curr_step>0:
        print("----------- trainable parameters --------------")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        print("-----------------------------------------------")
    
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[opts.gpu_ids[args.local_rank]], find_unused_parameters=True)
    model.train()
   
    dataset_dict = get_dataset(opts)
    train_sampler = DistributedSampler(dataset_dict['train'], shuffle=True)
    
    train_loader = data.DataLoader(
        dataset_dict['train'], 
        batch_size=opts.dataset.batch_size,
        sampler=train_sampler,  
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True, 
        prefetch_factor=4)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.dataset.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.dataset.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if args.local_rank==0:
        print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
        (opts.dataset.name, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    max_iters = opts.train.train_epochs * len(train_loader)
    val_interval = max(100, max_iters // 10)
    metrics = StreamSegMetrics(sum(opts.num_classes), dataset=opts.dataset.name)

    train_sampler.set_epoch(0)
            
    if args.local_rank==0:
        print(f"... train epoch : {opts.train.train_epochs} , iterations : {max_iters} , val_interval : {val_interval}")
    # Create a GradScaler for automatic mixed precision (AMP) training
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    # Set up the loss function based on the configuration
    if opts.train.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=opts.dataset.ignore_index, 
                                                           reduction='mean')
    elif opts.train.loss_type == 'ce_loss':                                                                                                                                                                                                                                                                       
        criterion = torch.nn.CrossEntropyLoss(ignore_index=opts.dataset.ignore_index, reduction='mean')
    
    # Set up additional loss functions for MBS if enabled
    if opts.train.MBS == True:
        # Separating Background-Class - output distillation, orthogonal loss
        od_loss = utils.LabelGuidedOutputDistillation(reduction="mean", alpha=1.0).to(device)
        ortho_loss = utils.OtrthogonalLoss(reduction="mean", classes=target_cls).to(device)
    else:
        od_loss = utils.KnowledgeDistillationLoss(reduction="mean", alpha=1.0).to(device)
        ortho_loss = None
    # Adaptive Feature Distillation
    fd_loss = utils.AdaptiveFeatureDistillation(reduction="mean", alpha=1).to(device)

    criterion = criterion.to(device)
    cur_epochs = 0
    avg_loss = AverageMeter()
    
    for n_iter in range(max_iters):
        try:
            inputs, labels, _ = next(train_loader_iter)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            inputs, labels, _ = next(train_loader_iter)
            cur_epochs += 1
        
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        origin_labels = labels.clone()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=opts.amp):
            outputs, patches, cls_seg_feat, cls_token = model(inputs)
            lod = torch.zeros(1).to(device)
            lfd_patches = torch.zeros(1).to(device)
            lfd = torch.zeros(1).to(device)
            
            if opts.curr_step > 0:
                with torch.no_grad():
                    outputs_prev, patches_prev, cls_seg_feat_prev, _ = model_prev(inputs)
                    if opts.train.loss_type == 'bce_loss':
                        pred_prob = torch.sigmoid(outputs_prev).detach()
                    else:
                        pred_prob = torch.softmax(outputs_prev, 1).detach()
                
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                labels = torch.where((labels <= bg_label) & (pred_labels > bg_label) & (pred_scores >= opts.train.pseudo_thresh), 
                                        pred_labels, 
                                        labels)
                
                if opts.train.MBS:
                    object_scores = torch.zeros(pred_prob.shape[0], 2, pred_prob.shape[2], pred_prob.shape[3]).to(device)
                    object_scores[:, 0] = pred_prob[:, 0]
                    object_scores[:, 1] = torch.sum(pred_prob[:, 1:], dim=1)
                    labels = torch.where((labels == 0) & (object_scores[:, 0] < object_scores[:, 1]), 
                                                opts.dataset.ignore_index, 
                                                labels)
                    
                if opts.train.MBS:
                    with torch.no_grad():
                        mask_origin = model_prev.get_masks()
                    HW = int(math.sqrt(patches.shape[1]))
                    label_temp = F.interpolate(labels.unsqueeze(1).float(), size=(HW, HW), mode='nearest').squeeze(1)
                    pred_score_mask = utils.make_scoremap(mask_origin, label_temp, target_cls, bg_label, ignore_index=opts.dataset.ignore_index)
                    pred_scoremap = pred_score_mask.squeeze().reshape(-1, HW*HW)
                    lfd_patches = fd_loss(patches.unsqueeze(1), patches_prev.unsqueeze(1), weights=pred_scoremap.unsqueeze(-1).unsqueeze(1))
                else:
                    lfd_patches = fd_loss(patches, patches_prev, weights=1)
                    
                lfd = lfd_patches + fd_loss(cls_seg_feat[:,:-len(target_cls)], cls_seg_feat_prev, weights=1)

                if opts.train.MBS:
                    lod = od_loss(outputs, outputs_prev, origin_labels) * opts.train.distill_args + ortho_loss(cls_token, weight=opts.num_classes[-1]/sum(opts.num_classes))
                else:
                    lod = od_loss(outputs, outputs_prev) * opts.train.distill_args      
                
            seg_loss = criterion(outputs, labels.type(torch.long))
            
            loss_total = seg_loss + lfd + lod
                
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        avg_loss.update(loss_total.item())
        scaler.update()
        
        if (n_iter+1) % opts.train.log_iters == 0 and args.local_rank==0:
            delta, eta = cal_eta(time0, n_iter+1, max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("[Epochs: %d Iter: %d] Elasped: %s; ETA: %s; LR: %.3e; loss: %f; FD_loss: %f; OD_loss: %f"%(cur_epochs, n_iter+1, delta, eta, lr, avg_loss.avg, lfd.item(), 
                                                                                                                                 lod.item()))
            writer.add_scalar(f'loss/train_{opts.curr_step}', loss_total.item(), n_iter+1)
            writer.add_scalar(f'lr/train_{opts.curr_step}', lr, n_iter+1)
            record_inputs, record_outputs, record_labels = imutils.tensorboard_image(inputs=inputs, outputs=outputs, labels=labels, dataset=opts.dataset.name)
            
            writer.add_image(f"input/train_{opts.curr_step}", record_inputs, n_iter+1)
            writer.add_image(f"output/train_{opts.curr_step}", record_outputs, n_iter+1)
            writer.add_image(f"label/train_{opts.curr_step}", record_labels, n_iter+1)
            
        if (n_iter+1) % val_interval == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            model.eval()
            val_score = validate(opts=opts, model=model, loader=val_loader, 
                              device=device, metrics=metrics)
            
            if args.local_rank==0:
                logging.info(metrics.to_str(val_score))
            model.train()
              
            writer.add_scalars(f'val/train_{opts.curr_step}', {"Overall Acc": val_score["Overall Acc"],
                                            "Mean Acc": val_score["Mean Acc"],
                                            "Mean IoU": val_score["Mean IoU"]}, n_iter+1)
            class_iou = list(val_score['Class IoU'].values())
            curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
            if args.local_rank==0:
                print("curr_val_score : %.4f" % (curr_score))
            
            if curr_score > best_score and args.local_rank==0:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step))
                
    if args.local_rank==0:            
        print("... Training Done")
    time.sleep(2)
    
    if opts.curr_step >= 0:
        if args.local_rank==0:
            logging.info("... Testing Best Model")
        best_ckpt = ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))["model_state"]
        model.module.load_state_dict(checkpoint, strict=True)
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        if args.local_rank==0:
            logging.info(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(opts.dataset.name, opts.task, 0))
        
        if args.local_rank==0:
            logging.info(f"...from 1 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[1:first_cls]))
            logging.info(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
            logging.info(f"...from 1 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[1:first_cls]))
            logging.info(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))            
            

if __name__ == "__main__":
    
    args = parser.parse_args()
    opts = OmegaConf.load(args.config)
    random.seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    opts.dataset.batch_size = opts.dataset.batch_size // len(opts.gpu_ids)

    if args.local_rank == 0:
        setup_logger(filename=str(args.log)+'.log')
        logging.info('\nconfigs: %s' % opts)

    start_step = opts.curr_step
    total_step = len(get_tasks(opts.dataset.name, opts.task))

    torch.cuda.set_device(opts.gpu_ids[args.local_rank])
    dist.init_process_group(backend=args.backend,)
    for step in range(start_step, total_step):
        opts.curr_step = step
        train(opts=opts)