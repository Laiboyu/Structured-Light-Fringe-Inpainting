import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.config import Config
from src.dataset import Dataset, Random_Rescale_Rotate
from src.models import ContourModel, InpaintingModel
from src.utils import create_dir, stitch_images
from src.metrics import EdgeAccuracy, Dice, MIoU
from torchmetrics import MeanSquaredError

class Structured_Light_Inpaint():
    
    def __init__(self, config):
        # 設定class的初始設定
        self.config = config
        self.model_name = config.NAME
        # 設定輪廓連家模塊網路-ContourModel以及條紋修補模塊網路-InpaintingModel
        self.contour_model = ContourModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        # 定義網路所使用的Metrics
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        self.dice = Dice().to(config.DEVICE)
        self.miou = MIoU().to(config.DEVICE)
        self.mse = MeanSquaredError().to(config.DEVICE)
        # 定義資料集的flist讀取路徑(利用config檔進行定義)
        config.TRAIN_FO_FLIST = os.path.join(config.TRAIN_FLIST_PATH, 'First_Order.flist')
        config.TRAIN_GT_FLIST = os.path.join(config.TRAIN_FLIST_PATH, 'Ground_Truth.flist')
        config.TRAIN_FLIST    = os.path.join(config.TRAIN_FLIST_PATH, 'Normalize_Image.flist')
        
        config.VAL_FO_FLIST = os.path.join(config.VAL_FLIST_PATH, 'First_Order.flist')
        config.VAL_GT_FLIST = os.path.join(config.VAL_FLIST_PATH, 'Ground_Truth.flist')
        config.VAL_FLIST    = os.path.join(config.VAL_FLIST_PATH, 'Normalize_Image.flist')
        
        # 定義訓練資料及以及驗證資料集 
        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_GT_FLIST, config.TRAIN_FO_FLIST,
                                     transform=transforms.Compose(
                                              [Random_Rescale_Rotate(image_size=config.INPUT_SIZE,
                                                                     angle_range=[0, 360],
                                                                     scale_ratio=[0.8, 1.2])])
                                     )
        
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_GT_FLIST, config.VAL_FO_FLIST, 
                                   transform=transforms.Compose(
                                            [Random_Rescale_Rotate(image_size=config.INPUT_SIZE,
                                                                   angle_range=[0, 360], 
                                                                   scale_ratio=[0.8, 1.2])])
                                   )
        
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
            
        # definition of path
        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')
        

    def load(self):

        self.contour_model.load()
        self.inpaint_model.load()

    def save(self):

        self.contour_model.save()
        self.inpaint_model.save()
        
    # 定義train()function去訓練網路
    def train(self):
        # 使用DataLoader()將train_dataset以batch_size的大小打亂資料排序(shuffle)成train_loader
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        total = len(train_loader)
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        # 定義tensorboard物件(writer)來進行指標計算追蹤
        writer = SummaryWriter(self.config.TENSORBOARD)
        # 利用config檔設定訓練的總epoch數量
        epoch_num = self.config.EPOCH

        # 開始訓練步驟
        for epoch in range(0, epoch_num):
            
            print('====================================================================')
            print('Training epoch: %d' % epoch)
            print('====================================================================')
            print('start training...\n')
            
            # 初始化驗證指標的數值
            precision_sum = 0
            recall_sum = 0
            dice_sum = 0
            miou_sum = 0
            mse_sum = 0
            
            for index, items in enumerate(train_loader):
                
                index += 1 

                self.contour_model.train()
                self.inpaint_model.train()
                
                # 將特徵影像放入gpu進行加速訓練
                fo_images, images, edges, gt_images, gt_edges = self.cuda(*items)
                
                # Contour-Connect所修補完成的影像c_outputs替代原始破損的條紋輪廓影像edges作為Fringe-Inpainting的輸入來進行增強
                c_outputs, c_logs = self.contour_model.process(fo_images, images, edges, gt_images, gt_edges, train=True)
                # Fringe-Inpainting取得修補完成的影像i_outputs(model.process中的初始定義train=None)
                i_outputs, i_logs = self.inpaint_model.process(fo_images, images, c_outputs.detach(), gt_images, gt_edges, train=True)

                # Metrics的數值計算
                c_precision, c_recall = self.edgeacc(gt_edges, c_outputs)
                c_logs.append(('Contour_Precision', c_precision.item()))
                c_logs.append(('Contour_Recall', c_recall.item()))

                i_dice = self.dice(gt_images, i_outputs)
                i_miou = self.miou(gt_images, i_outputs)
                i_mse = self.mse(gt_images, i_outputs)
                i_logs.append(('Inpainting_Dice', i_dice.item()))
                i_logs.append(('Inpainting_MIoU', i_miou.item()))
                i_logs.append(('Inpainting_MSE', i_mse.item()))
                
                # 將Metrics的計算成果存取成log檔

                precision_sum += c_precision.item()
                recall_sum += c_recall.item()
                dice_sum += i_dice.item()
                miou_sum += i_miou.item()
                mse_sum += i_mse.item()
                
                # logs為各個model在訓練過程中每一次batch所計算各個模塊網路的損失函數數值
                logs = c_logs + i_logs

                iteration = self.inpaint_model.iteration

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs
                
                # 每一次LOG_INTERVAL的區間去記錄下模型表現
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)
                    
                # 下述程式為監控網路的訓練過程百分比
                print('\r' + '[Training]:[%s%s]%.2f%%;' % ('█' * int(index*20/total), ' ' * (20-int(index*20/total)),
                                    float(index/total*100)), end=' ') 
            print('\n')

            train_logs = [
                ("Contour_Precision",  precision_sum/total),
                ("Contour_Recall",     recall_sum/total),
                ("Inpainting_Dice",    dice_sum/total),
                ("Inpainting_MIoU",    miou_sum/total),
                ("Inpainting_MSE",     mse_sum/total),
            ]
            
            # 每一次epoch訓練結束將平均計算的指標數值print()出
            template = "[Train]- Contour_Precision:{}, Contour_Recall:{}, Inpainting_Dice:{}, Inpainting_MIoU:{}, Inpainting_MSE:{}"
            print(template.format(
                                "{:.3f}".format(train_logs[0][1]),
                                "{:.3f}".format(train_logs[1][1]),
                                "{:.3f}".format(train_logs[2][1]),
                                "{:.3f}".format(train_logs[3][1]),
                                "{:.3f}".format(train_logs[4][1]),
                                ))

            print('--------------------------------------------------------------------')
            print('start validating ...\n')
            
            # 去計算驗證數據集的指標計算數值
            eval_logs = self.eval()
            
            # 轉換list物件到dictionary物件
            train_dict = self.convert_list_to_dictonary(train_logs)
            eval_dict  = self.convert_list_to_dictonary(eval_logs)
            
            iteration_order =  iteration / len(train_loader)
            print('iteration_order', iteration_order)
            
            # 利用 Tensorboard 去紀錄每一個epoch的模型表現成果
            writer.add_scalars('Contour_Precision', {'Train': train_dict['Contour_Precision'], 'Validation': eval_dict['Contour_Precision']}, iteration_order)
            writer.add_scalars('Contour_Recall',    {'Train': train_dict['Contour_Recall'],    'Validation': eval_dict['Contour_Recall']}, iteration_order)
            writer.add_scalars('Inpainting_Dice',   {'Train': train_dict['Inpainting_Dice'],   'Validation': eval_dict['Inpainting_Dice']}, iteration_order)
            writer.add_scalars('Inpainting_MIoU',   {'Train': train_dict['Inpainting_MIoU'],   'Validation': eval_dict['Inpainting_MIoU']}, iteration_order)   
            writer.add_scalars('Inpainting_MSE',    {'Train': train_dict['Inpainting_MSE'],    'Validation': eval_dict['Inpainting_MSE']}, iteration_order)
            
            print('\nstart sampling and saving model...\n')
            
            # sample()從驗證資料集中隨機取sample_size的影像丟入網路模型中去看修補成果
            self.sample()
            # save()保存網路模型的權重值
            self.save()
        

    # 定義eval()function去計算驗證數據集的指標計算數值
    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )
        
        total = len(self.val_dataset)

        self.contour_model.eval()
        self.inpaint_model.eval()

        eval_recall = 0
        eval_precision = 0
        eval_dice = 0
        eval_miou = 0
        eval_mse  = 0
        
        # with torch.no_grad()代表不將特徵影像放入back propagation中避免在驗證步驟中更新到網路權重值
        with torch.no_grad():
            for index, items in enumerate(val_loader):
                index += 1

                fo_images, images, edges, gt_images, gt_edges = self.cuda(*items)
                
                c_outputs, c_logs = self.contour_model.process(fo_images, images, edges, gt_images, gt_edges, train=None)
                i_outputs, i_logs = self.inpaint_model.process(fo_images, images, c_outputs.detach(), gt_images, gt_edges, train=None)
                  
                c_precision, c_recall = self.edgeacc(gt_edges, c_outputs)
                c_logs.append(('Contour_Precision', c_precision.item()))
                c_logs.append(('Contour_Recall', c_recall.item()))

                i_dice = self.dice(gt_images, i_outputs)
                i_miou = self.miou(gt_images, i_outputs)
                i_mse = self.mse(gt_images, i_outputs)
                i_logs.append(('Inpainting_Dice', i_dice.item()))
                i_logs.append(('Inpainting_MIoU', i_miou.item()))
                i_logs.append(('Inpainting_MSE', i_mse.item()))
                
                eval_precision += c_precision.item()
                eval_recall    += c_recall.item()
                eval_dice      += i_dice.item()
                eval_miou      += i_miou.item()
                eval_mse       += i_mse.item()
                
                logs = c_logs + i_logs
                
                print('\r' + '[Validating]:[%s%s]%.2f%%;' % ('█' * int(index*20/total), ' ' * (20-int(index*20/total)),
                        float(index/total*100)), end=' ') 

            print('\n')
                
            eval_logs = [
                ("Contour_Precision",  eval_precision/index),
                ("Contour_Recall",     eval_recall/index),
                ("Inpainting_Dice", eval_dice/index),
                ("Inpainting_MIoU", eval_miou/index),
                ("Inpainting_MSE",  eval_mse/index),
            ]
            
            template = "[Validation]- Contour_Precision:{}, Contour_Recall:{}, Inpainting_Dice:{}, Inpainting_MIoU:{}, Inpainting_MSE:{}"
            print(template.format(
                                "{:.3f}".format(eval_logs[0][1]),
                                "{:.3f}".format(eval_logs[1][1]),
                                "{:.3f}".format(eval_logs[2][1]),
                                "{:.3f}".format(eval_logs[3][1]),
                                "{:.3f}".format(eval_logs[4][1]),
                                ))
            
        
        return eval_logs

    def sample(self, it=None):

        if len(self.val_dataset) == 0:
            return
        
        device = 'cpu'
        self.contour_model.to(device)
        self.contour_model.eval()
        
        self.inpaint_model.to(device)
        self.inpaint_model.eval()

        items = next(self.sample_iterator)
        
        fo_images, images, edges, gt_images, gt_edges = items
        
        iteration = self.inpaint_model.iteration
        # 如果沒有要進行訓練以及驗證，直接將輸入特徵影像放入self.model()獲取模型輸出
        c_outputs = self.contour_model(fo_images, images, edges).detach()
        i_outputs = self.inpaint_model(fo_images, images, c_outputs)

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(fo_images),
            self.postprocess(images),
            self.postprocess(edges),
            self.postprocess(c_outputs),
            self.postprocess(gt_images),
            self.postprocess(gt_edges),
            self.postprocess(i_outputs),
            img_per_row = image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('saving sample ' + name)
        images.save(name)
        
        self.contour_model.cuda()
        self.inpaint_model.cuda()

    # 定義log()去進行文件的撰寫
    def log(self, logs):
        with open(self.log_file, 'a') as f:
            for item in logs:
                f.write(str(item[0]))
                f.write(': ')
                f.write(str(item[1]))
                f.write(' ')
            f.write('\n')
            
    # 定義cuda()去指定影像用gpu計算
    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)
    
    
    # 定義postprocess()將正規化影像*255來進行影像顯示與儲存
    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
    
    def convert_list_to_dictonary(self, tuple_list):
        tuple_dict = {}
        for log in tuple_list:
            tuple_dict[log[0]] = log[1]
        return tuple_dict

def train():
    '''
    首先定義Structured_Light_Inpaint class物件來進行整體系統架構的建立，
    利用Structured_Light_Inpaint.train()來進行訓練資料庫建立以及網路訓練，
    在每一次Epoch訓練完成後，再使用Structured_Light_Inpaint.eval()來建立驗證資料集以及驗證指標數值計算。
    '''
    # 使用config檔來讀取網路設定參數值以及dataset
    config_path = '.\checkpoints\config.yml'
    config = Config(config_path)
    config.print()
    
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    
    # 設定網路運算為 'gpu' or 'cpu'
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")
    
    # 設定初始的隨機因素
    cv2.setNumThreads(0)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
        
    # build the model and initialize
    model = Structured_Light_Inpaint(config)
    model.train()

if __name__ == "__main__":
    train()