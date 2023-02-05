import os
import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import Config
from src.models import ContourModel, InpaintingModel
from src.dataset import Dataset, Prediction_Dataset
from src.metrics import EdgeAccuracy, Dice, MIoU
from torchmetrics import MeanSquaredError

def test():
    '''
    定義test function()來進行測試資料集的讀取以及指標計算，
    首先利用Dataset()讀取資料，再利用Metrics進行二元影像指標計算
    '''
    ## test
    # 使用config檔來讀取網路設定參數值以及dataset flist的路徑
    config_path = './checkpoints\config.yml'
    config = Config(config_path)
    config.print()
    
    # init device
    # 定義網路的設備為"gpu"or"cpu"
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")
    # config.DEVICE = torch.device("cpu")
    
    # 讀取訓練完成的網路權重值
    contour_model = ContourModel(config).to(config.DEVICE)
    contour_model.load()
    inpainting_model = InpaintingModel(config).to(config.DEVICE)
    inpainting_model.load()
    
    print('---------------------------------\nstart testing...\n')
    
    # Metrics
    # 定義網路metrics物件
    Edgeacc_Metric = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
    Dice_Metric = Dice().to(config.DEVICE)
    MIoU_Metric = MIoU().to(config.DEVICE)
    MAE = MeanSquaredError().to(config.DEVICE)
        
    TEST_FO_FLIST = os.path.join(config.TEST_FLIST_PATH, 'First_Order.flist')
    TEST_GT_FLIST = os.path.join(config.TEST_FLIST_PATH, 'Ground_Truth.flist')
    TEST_FLIST    = os.path.join(config.TEST_FLIST_PATH, 'Normalize_Image.flist')
    
    
    test_dataset = Dataset(config, TEST_FLIST, TEST_GT_FLIST, TEST_FO_FLIST)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 定義存放預測資料夾物徑
    PREDICTIONDIR = os.path.join(config.TEST_FLIST_PATH, 'Prediction', 'Prediction')
    COMPARISONDIR = os.path.join(config.TEST_FLIST_PATH, 'Prediction', 'Comparison')
    
    os.makedirs(PREDICTIONDIR)
    os.makedirs(COMPARISONDIR)
    
    recall_SUM = 0
    precision_SUM = 0
    dice_SUM = 0 
    miou_SUM = 0
    mse_SUM = 0
    
    total = len(test_loader)
    with torch.no_grad():
        for index, items in enumerate(test_loader):
            
            # 將輸入影像fo_images, images, edges放入contour_model進行修補
            # 接著使用fo_images, images, c_outputs作為輸入放入fringe-inpainting進行修補
            fo_images, images, edges, gt_images, gt_edges = (item.cuda() for item in items)  
            c_outputs = contour_model(fo_images, images, edges).detach()
            i_outputs = inpainting_model(fo_images, images, c_outputs)
                
            # Metric Values Calculation
            # 將生成影像與gt_images, gt_edges等基礎事實影像進行指標運算
            precision, recall = Edgeacc_Metric(gt_images, i_outputs.detach())
            dice = Dice_Metric(gt_images, i_outputs.detach())
            miou = MIoU_Metric(gt_images, i_outputs.detach())
            MSE = MAE(gt_images, i_outputs.detach())
            
            precision_SUM += precision.item()
            recall_SUM += recall.item()
            dice_SUM += dice.item()
            miou_SUM += miou.item()
            mse_SUM += MSE.item()
            
            #　設定字典物件存放輸入影像以及生成影像特徵
            img_dictionary = {"First Order":fo_images[0], "Image Input":images[0], "Edge Input":edges[0], 
                              "Contour Connect":c_outputs[0].detach(), "Fringe Inpainting":i_outputs[0].detach()}
             
            
            # 利用plt套件提取影像字典物件來顯示模型修補成果
            fig = plt.figure(figsize=(60, 30))
            for i, img_name in enumerate(img_dictionary):
                ax = fig.add_subplot(1, len(img_dictionary), i+1)
                ax.imshow(transforms.ToPILImage()(img_dictionary[img_name]), cmap='gray')
                plt.title(img_name, fontsize=80)
                
            plt.pause(0.00001)
            
            index += 1
            fig.savefig(os.path.join(COMPARISONDIR, 'cam_'+str(index).rjust(2,'0')+'.png'))
            
            result_path = os.path.join(PREDICTIONDIR, 'cam_'+str(index).rjust(2,'0')+'.png')
            # 利用torchvision.utils.save_image函式存取tensor影像
            torchvision.utils.save_image(i_outputs[0].detach(), result_path)
    
            print('\r' + '[Testing]:[%s%s]%.2f%%;' % ('█' * int(index*20/total), ' ' * (20-int(index*20/total)),
                                        float(index/total*100)), end=' ') 

    print('\n')
    
    # print模型對於指標的平均表現成果
    template = "[Test]- Precision:{}, Recall:{}, Dice:{}, MIoU:{}, MSE:{}"
    print(template.format(
                        "{:.3f}".format(precision_SUM/index),
                        "{:.3f}".format(recall_SUM/index),
                        "{:.3f}".format(dice_SUM/index),
                        "{:.3f}".format(miou_SUM/index),
                        "{:.3f}".format(mse_SUM/index),
                        ))
    
    metrics_txt_path = os.path.join(config.TEST_FLIST_PATH, 'Metrics.txt')
    
    # 將指標的平均表現成果寫入成txt檔進行紀錄
    with open(metrics_txt_path, 'w') as fp:
          
        fp.write('Precision : {:3f}\n'.format(precision_SUM/index))
        fp.write('Recall :    {:3f}\n'.format(recall_SUM/index))
        fp.write('Dice :      {:3f}\n'.format(dice_SUM/index))
        fp.write('MIoU :      {:3f}\n'.format(miou_SUM/index))
        fp.write('MSE :       {:3f}\n'.format(mse_SUM/index))
        

        
if __name__ == "__main__":
    test()
        