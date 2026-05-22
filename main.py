from ast import arg
import os
import argparse
import random
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from Dataset.dataset import get_loader, test_dataset
from utils.utils import clip_gradient, Init_Log, AvgMeter
from utils.metric import BinaryMetricRecorder
from lib.build_model import building_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="GroupD_Mix",help='choise differents seg model')
parser.add_argument('--dataset', type=str, default="BUSI", help='four dataset for BUSI,TN3K,MMOTU,DDTI')
parser.add_argument('--epoch', type=int,default=300, help='epoch number')
parser.add_argument('--init_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--iter_num', type=int, default=0, help='segmentation network learning rate')
parser.add_argument('--Deep_Supervision',action='store_true', help='use deep supervision')
parser.add_argument('--augmentation',default=True, help='choose to do train data aug')
parser.add_argument('--batchsize', type=int,default=8, help='training batch size')
parser.add_argument('--img_size', type=int, default=224, help='img size of train')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
parser.add_argument('--early_stopping', default=-1, type=int,metavar='N', help='early stopping (default: -1)for save time and GPU!')
parser.add_argument('--flod', type=str, default="1", help='cross fiel number')
parser.add_argument('--seed', type=int, default=41, help='random seed')
args = parser.parse_args()

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def dice_bce_loss(input, target):
    bce = F.binary_cross_entropy_with_logits(input, target)
    smooth = 1e-5
    input = torch.sigmoid(input)
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return 0.5 * bce + dice

def test(model, val_dir, logger, args):
    
    metrics_v2 = BinaryMetricRecorder(metric_names=BinaryMetricRecorder.suppoted_metrics)
    test_loader = test_dataset(dataset=args.dataset, 
                               test_file_dir=val_dir,  
                               testsize=args.img_size)
    global best_iou
    global stop 
    model.eval()
    _, gt = test_loader.get_list()

    for i in tqdm(range(len(gt)), total=len(gt),desc='Eval...',ascii=True):
        image, gt, _ = test_loader.load_data()
        gt = np.asarray(gt, np.uint8)

        image = image.cuda()

        if args.Deep_Supervision :
            out, _, _, _  = model(image)
            res = F.interpolate(out, size=gt.shape, mode='bilinear', align_corners=False)
        else:
            out  = model(image)
            res = F.interpolate(out  , size=gt.shape, mode='bilinear', align_corners=False)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = np.asarray(res*255, np.uint8)
        target = np.array(gt)
   
        metrics_v2.step(input, target)

    
    m_IoU_v2,m_Dice_v2,m_Acc_v2,m_Pre_v2,sm_v2,wfm_v2,m_Pec_v2,m_Fm_v2= metrics_v2.show()["sample_biiou"], \
        metrics_v2.show()["sample_bidice"], metrics_v2.show()["sample_bioa"], metrics_v2.show()["sample_bipre"],\
        metrics_v2.show()["sm"],metrics_v2.show()["wfm"], metrics_v2.show()["sample_bispec"], metrics_v2.show()["sample_bifm"]
    
    logger.info("=========> Evaluate")
    logger.info(f"==> mIoU: {m_IoU_v2} mDice: {m_Dice_v2} Acc: {m_Acc_v2} mPre {m_Pre_v2}")
    logger.info(f"==> sm: {sm_v2} wfm:{wfm_v2} mPec: {m_Pec_v2} mFm {m_Fm_v2}")

    is_best = m_IoU_v2 > best_iou # bool type
    best_iou = max(m_IoU_v2, best_iou)
    if is_best :
        if not os.path.isdir("./checkpoint"):
            os.makedirs("./checkpoint")
        if args.dataset == "BUSI" :
            torch.save(model.state_dict(), f'checkpoint/{args.model}_{args.dataset}_model7_{str(args.flod)}.pth')  
        logger.info("=========> Saved best model\n")
        stop = 0
    
    return m_IoU_v2



def train(train_loader, model, optimizer, epoch,logger, args):
    model.train()
    
    global best_iou
    loss_record = AvgMeter()
    total_step = len(train_loader)
    max_iterations = total_step * args.epoch
    for i, pack in tqdm(enumerate(train_loader, start=1),total=total_step,desc='Training...',ascii=True):

        optimizer.zero_grad()

        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        if args.Deep_Supervision :
            out, out4, out_3, out_2 = model(images)
            loss_1 = dice_bce_loss(out, gts)
            loss_4 = dice_bce_loss(out4, gts)
            loss_3 = dice_bce_loss(out_3, gts)
            loss_2 = dice_bce_loss(out_2, gts)
            loss = loss_1 + 0.45*loss_4 + 0.45*loss_3 + 0.45*loss_2
        else:
            out = model(images)
            loss_1 = dice_bce_loss(out, gts)
            loss = loss_1

        lr_ = args.init_lr * (1.0 - args.iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
            
        loss.backward()
        clip_gradient(optimizer, args.clip)
        optimizer.step()
            
        args.iter_num = args.iter_num + 1
        loss_record.update(loss.data, args.batchsize)
 
    logger.info('=========> Epoch[{}|{}] Train_Total_Loss:{:.4f} LR: {:.5f} ' 
                'Current Best IoU {}'.format(epoch, args.epoch, 
                                          loss_record.show(), optimizer.param_groups[0]['lr'],
                                          best_iou))

def main(args):
    
    t = datetime.now()
    tt  = t.strftime('%Y-%m-%d-%H:%M:%S' )
    
    save_log = 'log/{}/'.format(args.model)
    if not os.path.exists(save_log):
            os.makedirs(save_log)
    if args.dataset == "BUSI": # Dataset/BUSI Dataset/BUSI/BUSI_test_1.txt
        train_dir = os.path.join('Dataset', args.dataset, args.dataset+"_train_"+str(args.flod)+".txt")
        val_dir = os.path.join('Dataset', args.dataset, args.dataset+"_test_"+str(args.flod)+".txt")
        logger_name =  save_log + args.model + '_' + args.dataset +'_'+ tt +'_'+ str(args.flod) +'_training.log'
    
    logger = Init_Log(logger_name)
    seed_torch(args.seed)
    model = building_model(args, parser)
    optimizer = torch.optim.SGD(model.parameters(), args.init_lr, weight_decay=1e-4, momentum=0.9)
    
    train_loader = get_loader(dataset=args.dataset, 
                              train_file_dir=train_dir,resize=args.img_size, 
                              batchsize=args.batchsize,
                              augmentation=args.augmentation)
    
    
    
    logger.info("=========> Experimental INFO:")
    logger.info(f"=========> Model: {args.model}")
    logger.info(f"=========> Dataset: {args.dataset}, Data_dir: {train_dir}")
    logger.info(f"=========> Deep Supervision: {args.Deep_Supervision}")
    logger.info(f"=========> Training Size: {args.img_size}")
    logger.info("=========> Training Start:")
    global stop
    for epoch in range(1, args.epoch):
        train(train_loader, model, optimizer, epoch, logger,args)
        test(model, val_dir,logger,args)
        stop += 1
        if args.early_stopping >= 0 and stop  >= args.early_stopping:
            logger.info(f"After {args.early_stopping} training epochs have no new best iou, Early Stopping at Epoch: {epoch}")
            break

if __name__ == '__main__':
    best_iou = 0.0
    stop = 0
    main(args)
    # pip install pysodmetrics==1.4.2
    # python main.py --model GroupD_Mix --Deep_Supervision --flod 1 --early_stopping 70
