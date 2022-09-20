import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from src.config import Config
from src.models import ContourModel, InpaintingModel
from src.dataset import Dataset, Prediction_Dataset

def predict():
    '''
    將四十張的編碼影像作為輸入，在config檔裡設定資料夾路徑，
    執行predict()儲存修補完成的影像。
    '''
    # predict
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
    
    # 讀取訓練完成的網路權重值
    contour_model = ContourModel(config).to(config.DEVICE)
    contour_model.load()
    inpainting_model = InpaintingModel(config).to(config.DEVICE)
    inpainting_model.load()
    
    print('start predicting...\n')
    
    TEST_FO_FLIST = os.path.join(config.PREDICT_FLIST_PATH, 'First_Order.flist')
    TEST_FLIST    = os.path.join(config.PREDICT_FLIST_PATH, 'Normalize_Image.flist')
    
    # 利用Prediction_Dataset來
    predict_dataset = Prediction_Dataset(config, TEST_FLIST, TEST_FO_FLIST)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
    
    # 定義存放預測資料夾物徑
    PREDICTIONDIR = os.path.join(config.PREDICT_FLIST_PATH, 'Prediction')
    os.makedirs(PREDICTIONDIR)

    total = len(predict_loader)
    with torch.no_grad():
        for index, items in enumerate(predict_loader):
            
            
            fo_images, images, edges = (item.cuda() for item in items)
            
            c_outputs = contour_model(fo_images, images, edges).detach()
            i_outputs = inpainting_model(fo_images, images, c_outputs)
            
            # 指定影像存取路徑，利用torchvision.utils.save_image儲存影像
            i_pred_result_path = os.path.join(PREDICTIONDIR, 'cam_'+str(index+1).rjust(2,'0')+'.png')
            torchvision.utils.save_image(i_outputs[0].detach(), i_pred_result_path)
                
            index += 1
            print('\r' + '[Prediction]:[%s%s]%.2f%%;' % ('█' * int(index*20/total), ' ' * (20-int(index*20/total)),
                                            float(index/total*100)), end=' ')
        print('\n')
     
if __name__ == "__main__":
    predict()
        