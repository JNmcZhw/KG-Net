import argparse
import time
import numpy as np
from data.metrics import psnr, ssim
import warnings
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch import optim, nn
from torch.utils.data import DataLoader
import utils
from data.dataloader import TrainDataloader, TestDataloader
from networks import TSNet2, VGG19CR
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Basic options
parser = argparse.ArgumentParser()
parser.add_argument('--clear_dir', type=str, default='./datasets/BeDDE/train/clear/',
                    help='Directory for saving clear images')
parser.add_argument('--hazy_dir', type=str, default='./datasets/BeDDE/train/haze/',
                    help='Directory for saving hazy images')
parser.add_argument('--category', type=str, default='BeDDE', help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--hazy_test_dir', type=str, default='./datasets/BeDDE/test/haze/',
                    help='Save a directory with hazy images for testing purposes')
parser.add_argument('--clear_test_dir', type=str, default='./datasets/BeDDE/test/clear/',
                    help='Save a directory with clear images for testing purposes')
parser.add_argument('--save_model_dir', default='./checkpoints/BeDDE/', help='Directory to save the model')
parser.add_argument('--save_val_dir', default='./results/val/', help='Directory to save the val results')
parser.add_argument('--save_good_model_dir', default='./checkpoints/Teacher/BeDDE_T.pth',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs', help='Directory to save the logs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, default=50)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--cr_weight', type=float, default=0.1)
parser.add_argument('--kt_weight', type=float, default=0.01)
parser.add_argument('--n_threads', type=int, default=0)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
    ]

    train_dataset = TrainDataloader(args.hazy_dir, args.clear_dir, transform=transforms_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_threads)

    val_sets = TestDataloader(args.hazy_test_dir, args.clear_test_dir, transform=transforms_train)
    val_loader = DataLoader(dataset=val_sets, batch_size=args.batch_size // args.batch_size, shuffle=False)

    dataset_length = len(train_loader)

    logger_train = utils.Logger(args.max_epoch, dataset_length)
    logger_val = utils.Logger(args.max_epoch, len(val_loader))

    T_S = TSNet2.DehazeNet().to(device)

    print('The models are initialized successfully!')

    T_S.train()

    opt_T_S = optim.Adam(T_S.parameters(), lr=args.lr, betas=(0.9, 0.999))

    lr_scheduler_T_S = torch.optim.lr_scheduler.LambdaLR(opt_T_S,
                                                         lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch,
                                                                                  args.decay_epoch).step)

    loss_l1 = nn.L1Loss().to(device)  # L1 loss
    loss_cr = VGG19CR.ContrastLoss().to(device)
    loss_kt = TSNet2.KTTeacher(T_path=args.save_good_model_dir).to(device)

    total_params = sum(p.numel() for p in T_S.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    max_ssim = 0
    max_psnr = 0
    all_ssims = []
    all_psnrs = []

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        ssims = []  # 每轮清空
        psnrs = []  # 每轮清空

        for i, batch in enumerate(train_loader):
            x = batch[0].to(device)  # selected hazy images
            clear = batch[1].to(device)  # clear images

            output = T_S(x)

            loss_L1 = loss_l1(output[0], clear) * args.l1_weight
            loss_CR = loss_cr(output[0], clear, x) * args.cr_weight
            loss_KT = loss_kt(output[1:], clear) * args.kt_weight

            loss = loss_L1 + loss_CR + loss_KT

            opt_T_S.zero_grad()
            loss.backward()
            opt_T_S.step()

            logger_train.log_train({
                # 'loss_l1': loss_L1,
                # 'loss_CLCR': loss_CLCR,
                # 'loss_CKT': loss_CKT,
                'loss': loss},
                images={'Hazy': x, 'Clear': clear, 'Output': output[0]})

        lr_scheduler_T_S.step()

        ################################################ Validating ##########################################

        with torch.no_grad():

            T_S.eval()

            torch.cuda.empty_cache()

            images_val = []
            images_name = []
            print("epoch:{}---> Metrics are being evaluated！".format(epoch))

            for a, batch_val in enumerate(val_loader):
                haze_val = batch_val[0].to(device)
                clear_val = batch_val[1].to(device)

                image_name = batch_val[2][0]

                output_val, _ = T_S(haze_val)

                images_val.append(output_val)
                images_name.append(image_name)

                psnr1 = psnr(output_val, clear_val)
                ssim1 = ssim(output_val, clear_val).item()

                psnrs.append(psnr1)
                ssims.append(ssim1)

                logger_val.log_val({'PSNR': psnr1,
                                    'SSIM': ssim1},
                                   images={'output_val': output_val, 'val': clear_val})

            psnr_eval = np.mean(psnrs)
            ssim_eval = np.mean(ssims)

            if psnr_eval > max_psnr:

                max_psnr = max(max_psnr, psnr_eval)

                torch.save(T_S.state_dict(), args.save_model_dir + args.category + "_Best_PSNR.pth")

                for i in range(len(images_name)):
                    torchvision.utils.save_image(images_val[i], args.save_val_dir + "{}".format(images_name[i]))

            if ssim_eval > max_ssim:
                max_ssim = max(max_ssim, ssim_eval)

                torch.save(T_S.state_dict(), args.save_model_dir + args.category + "_Best_SSIM.pth")
