import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, ContourGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
# from src.networks import InpaintGenerator, EdgeGenerator, Discriminator
# from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss

class BaseModel(nn.Module):
    '''
    定義初始的模型物件function來讀取、儲存模型權重
    '''
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        
        self.name = name
        self.config = config
        self.iteration = 0
        # 模塊網路的權重值路徑
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        # 讀取模塊網路的訓練權重(包含著generator以及discriminator)
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)
            
            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        # 儲存模塊網路的訓練權重以及逸代次數(包含著generator以及discriminator)
        print('saving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class ContourModel(BaseModel):
    """
    繼承BaseModel() class物件為基礎，來建立Contour-Connect模塊網路架構
    """
    def __init__(self, config):
        super(ContourModel, self).__init__('ContourModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = ContourGenerator(in_channels=3, use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        
        # 定義l1_loss以及adversarial_loss作為損失函數
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        
        # 加入優化器adam到網路架構中
        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
        
    def process(self, fo_images, images, edges, gt_images, gt_edges, train=None):
        # 開始訓練，將train設定為true，諾是沒有要進行訓練則設定為false
        
        if train == True:    
            self.iteration += 1
        outputs = self(fo_images, images, edges)
        
        # =====================================================================
        # Discriminator
        # =====================================================================
        # zero optimizers 初始化網路優化器的數值
        self.dis_optimizer.zero_grad()
        # discriminator loss
        #　對抗性的損失函數計算
        dis_loss = 0
        dis_input_real = torch.cat((gt_images, gt_edges), dim=1)
        dis_input_fake = torch.cat((gt_images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        # train = True時，進行網路權重值的反向梯度傳輸
        if train == True:
            if dis_loss is not None:
                dis_loss.backward()
            self.dis_optimizer.step()  
        
        # =====================================================================
        # Generator
        # =====================================================================
        # zero optimizers
        self.gen_optimizer.zero_grad()
        
        # generator adversarial loss
        gen_loss = 0
        gen_input_fake = torch.cat((gt_edges, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # generator feature matching loss
        # 取Discriminator中的每一層網路運算層進行l1-loss計算(gen_fake_featdis_real_feat)
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss
        
        if train == True:   
            if gen_loss is not None:
                gen_loss.backward()
            self.gen_optimizer.step()

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]
        
        return outputs, logs

    def forward(self, fo_images, images, edges):
        # 模塊網路(self)輸入以及輸出
        inputs = torch.cat((fo_images, images, edges), dim=1)
        outputs = self.generator(inputs)      
        return outputs


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator(in_channels=3)
        discriminator = Discriminator(in_channels=1, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, fo_images, images, edges, gt_images, gt_edges, train=None):
        
        if train == True:    
            self.iteration += 1
        outputs = self(fo_images, images, edges)
        
        # =====================================================================
        # Discriminator
        # =====================================================================
        # zero optimizers
        self.dis_optimizer.zero_grad()
        
        # discriminator loss
        dis_loss = 0
        dis_input_real = gt_images
        dis_input_fake = outputs.detach()
        dis_real, dis_real_feat = self.discriminator(dis_input_real)                    # in: [rgb(3)] SL-grayscale
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)                    # in: [rgb(3)] SL-grayscale
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        
        if train == True:
            dis_loss.backward()
            self.dis_optimizer.step()
            
        # =====================================================================
        # Generator
        # =====================================================================
        # zero optimizers
        self.gen_optimizer.zero_grad()
        
        # generator adversarial loss
        gen_loss = 0
        gen_input_fake = outputs
        # gen_input_fake = torch.cat((outputs, edges), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss
        
        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, gt_images) * self.config.L1_LOSS_WEIGHT
        gen_loss += gen_l1_loss
        
        if train == True:
            gen_loss.backward()
            self.gen_optimizer.step()

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l2", gen_l1_loss.item()),
        ]

        return outputs, logs

    def forward(self, fo_images, images, edges):
 
        inputs = torch.cat((fo_images, images, edges), dim=1)
        outputs = self.generator(inputs)      
        
        return outputs
