""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
# from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
# from visualize import plot


config = SearchConfig()
print(config.gpus)
device = torch.device("cuda")

# tensorboard
# writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
# writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
# config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    
    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model_1 = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                  net_crit, device_ids=config.gpus)
    
    torch.manual_seed(config.seed+1)
    torch.cuda.manual_seed_all(config.seed+1)
    model_2 = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                  net_crit, device_ids=config.gpus)

    torch.backends.cudnn.benchmark = True

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    # weights optimizer
    w_optim_1 = torch.optim.SGD(model_1.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim_1 = torch.optim.Adam(model_1.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)
    
    # weights optimizer
    w_optim_2 = torch.optim.SGD(model_2.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim_2 = torch.optim.Adam(model_2.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim_1, config.epochs, eta_min=config.w_lr_min)
    lr_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim_2, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model_1, model_2, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1_1 = 0.
    best_top1_2 = 0.
    for epoch in range(config.epochs):
        lr_scheduler_1.step()
        lr_1 = lr_scheduler_1.get_lr()[0]
        lr_scheduler_2.step()
        lr_2 = lr_scheduler_2.get_lr()[0]

        model_1.print_alphas(logger)
        model_2.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model_1, model_2, architect, w_optim_1, w_optim_2, alpha_optim_1, alpha_optim_2, lr_1, lr_2, epoch, config.lmbda)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1_1, top1_2 = validate(valid_loader, model_1, model_2, epoch, cur_step)

        # log
        # genotype
        genotype_1 = model_1.genotype()
        genotype_2 = model_2.genotype()
        logger.info("genotype_1 = {}".format(genotype_1))
        logger.info("genotype_2 = {}".format(genotype_2))

        # genotype as a image
        # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        # caption = "Epoch {}".format(epoch+1)
        # plot(genotype_1.normal, plot_path + "-normal", caption)
        # plot(genotype_1.reduce, plot_path + "-reduce", caption)
        # plot(genotype_2.normal, plot_path + "-normal", caption)
        # plot(genotype_2.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1_1 < top1_1:
            best_top1_1 = top1_1
            best_genotype_1 = genotype_1
            is_best_1 = True
        else:
            is_best_1 = False
            
        if best_top1_2 < top1_2:
            best_top1_2 = top1_2
            best_genotype_2 = genotype_2
            is_best_2 = True
        else:
            is_best_2 = False
            
        utils.save_checkpoint(model_1, config.path, 1, is_best_1)
        utils.save_checkpoint(model_2, config.path, 2, is_best_2)
        print("")

    logger.info("Final best Prec@1_1 = {:.4%}".format(best_top1_1))
    logger.info("Best Genotype_1 = {}".format(best_genotype_1))
    logger.info("Final best Prec@1_2 = {:.4%}".format(best_top1_2))
    logger.info("Best Genotype_2 = {}".format(best_genotype_2))


def train(train_loader, valid_loader, model_1, model_2, architect, w_optim_1, w_optim_2, alpha_optim_1, alpha_optim_2, lr_1, lr_2, epoch, lmbda):
    top1_1 = utils.AverageMeter()
    top5_1 = utils.AverageMeter()
    top1_2 = utils.AverageMeter()
    top5_2 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    # writer.add_scalar('train/lr', lr_1, cur_step)
    # writer.add_scalar('train/lr', lr_2, cur_step)

    model_1.train()
    model_2.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim_1.zero_grad()
        alpha_optim_2.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr_1, lr_2, w_optim_1, w_optim_2, lmbda)
        alpha_optim_1.step()
        alpha_optim_2.step()

        # phase 1. child network step (w)
        w_optim_1.zero_grad()
        w_optim_2.zero_grad()
        
        logits_1 = model_1(trn_X)
        pseudolabel_1 = torch.argmax(logits_1, dim=1)
        logits_2 = model_2(trn_X)
        pseudolabel_2 = torch.argmax(logits_2, dim=1)
        loss = model_1.criterion(logits_1, trn_y) + model_2.criterion(logits_2, trn_y) + lmbda * model_1.criterion(logits_1, pseudolabel_2) + lmbda * model_2.criterion(logits_2, pseudolabel_1)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model_1.weights(), config.w_grad_clip)
        nn.utils.clip_grad_norm_(model_2.weights(), config.w_grad_clip)
        w_optim_1.step()
        w_optim_2.step()

        prec1_1, prec5_1 = utils.accuracy(logits_1, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1_1.update(prec1_1.item(), N)
        top5_1.update(prec5_1.item(), N)
        
        prec1_2, prec5_2 = utils.accuracy(logits_2, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1_2.update(prec1_2.item(), N)
        top5_2.update(prec5_2.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1_1, top5=top5_1))
            
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1_2, top5=top5_2))

        # writer.add_scalar('train/loss', loss.item(), cur_step)
        # writer.add_scalar('train/top1', prec1_1.item(), cur_step)
        # writer.add_scalar('train/top5', prec5_1.item(), cur_step)
        # writer.add_scalar('train/top1', prec1_2.item(), cur_step)
        # writer.add_scalar('train/top5', prec5_2.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1_1.avg))
    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1_2.avg))


def validate(valid_loader, model_1, model_2, epoch, cur_step):
    top1_1 = utils.AverageMeter()
    top5_1 = utils.AverageMeter()
    top1_2 = utils.AverageMeter()
    top5_2 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model_1.eval()
    model_2.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits_1 = model_1(X)
            logits_2 = model_2(X)
            loss = model_1.criterion(logits_1, y) + model_2.criterion(logits_2, y)

            prec1_1, prec5_1 = utils.accuracy(logits_1, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1_1.update(prec1_1.item(), N)
            top5_1.update(prec5_1.item(), N)
            
            prec1_2, prec5_2 = utils.accuracy(logits_2, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1_2.update(prec1_2.item(), N)
            top5_2.update(prec5_2.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1_1, top5=top5_1))
                
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1_2, top5=top5_2))

    # writer.add_scalar('val/loss', losses.avg, cur_step)
    # writer.add_scalar('val/top1', top1_1.avg, cur_step)
    # writer.add_scalar('val/top5', top5_1.avg, cur_step)
    # writer.add_scalar('val/top1', top1_2.avg, cur_step)
    # writer.add_scalar('val/top5', top5_2.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1_1.avg))
    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1_2.avg))

    return top1_1.avg, top1_2.avg


if __name__ == "__main__":
    main()
