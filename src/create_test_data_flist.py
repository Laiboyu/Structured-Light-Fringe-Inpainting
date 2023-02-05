import os
import numpy as np

# 測試資料集路徑
rootdir = r'datasets\test_data\object_(2)'

# 分別按照3種不同的資料集進行讀取
for folder_name in ['First_Order', 'Normalize_Image', 'Ground_Truth']:
    
    path = os.path.join(rootdir, folder_name)
    output = os.path.join(rootdir, folder_name+'.flist')
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
    # 讀取影像路徑存取為list
    images = []
    for root, dirs, files in os.walk(path):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1].upper() in ext:
                images.append(os.path.join(root, file))
                
    # 儲存路徑到flist檔
    images = sorted(images)
    np.savetxt(output, images, fmt='%s')
