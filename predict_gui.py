import sys

import numpy
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from opts import get_parser
import torchvision.transforms as transforms
import torch
import cv2
from utils.dice_loss import get_soft_label


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 921)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 显示选择的图片
        self.img_width = 300
        self.img_height = 256
        self.img = None
        self.gt = None
        self.padding = 100
        self._translate = None
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10,110, self.img_width+self.padding, self.img_height+self.padding))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_img_show = QtWidgets.QLabel(self.frame)
        self.label_img_show.setGeometry(QtCore.QRect(10, 10, self.img_width+self.padding, self.img_height+self.padding))
        self.label_img_show.setObjectName("label_img_show")
        # self.label_img_show.setStyleSheet(("border:2px solid red"))
        # 显示照片对应GT
        self.GT_frame = QtWidgets.QFrame(self.centralwidget)
        self.GT_frame.setGeometry(QtCore.QRect(500, 110, self.img_width + self.padding, self.img_height + self.padding))
        self.GT_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.GT_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.GT_frame.setObjectName("detect_frame")
        self.GT_show = QtWidgets.QLabel(self.GT_frame)
        self.GT_show.setGeometry(QtCore.QRect(10, 10, self.img_width + self.padding, self.img_height + self.padding))
        self.GT_show.setObjectName("label_detect_show")

        # 显示预测结果
        self.detect_frame = QtWidgets.QFrame(self.centralwidget)
        self.detect_frame.setGeometry(QtCore.QRect(10, 500, self.img_width+self.padding, self.img_height+self.padding))
        self.detect_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.detect_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detect_frame.setObjectName("detect_frame")
        self.label_detect_show = QtWidgets.QLabel(self.detect_frame)
        self.label_detect_show.setGeometry(QtCore.QRect(10, 10, self.img_width+self.padding, self.img_height+self.padding))
        self.label_detect_show.setObjectName("label_detect_show")

        # self.info_show = QtWidgets.QLabel(self.detect_frame)
        # self.info_show.setGeometry(QtCore.QRect(10, -20, 100, 50))
        # self.info_show.setObjectName("word")
        # self.label_detect_show.setStyleSheet(("border:2px solid green"))

        # 显示dice score
        self.detect_frame = QtWidgets.QFrame(self.centralwidget)
        self.detect_frame.setGeometry(QtCore.QRect(500, 500, self.img_width + self.padding, self.img_height + self.padding))
        self.detect_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.detect_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detect_frame.setObjectName("detect_frame")
        self.dice_show = QtWidgets.QLabel(self.detect_frame)
        self.dice_show.setGeometry(QtCore.QRect(10, 10, self.img_width + self.padding, self.img_height + self.padding))
        self.dice_show.setObjectName("dice_show")

        # 按钮框架
        self.btn_frame = QtWidgets.QFrame(self.centralwidget)
        self.btn_frame.setGeometry(QtCore.QRect(10, 20, 1021, 80))
        self.btn_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.btn_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.btn_frame.setObjectName("frame_3")

        # 按钮水平布局
        self.widget = QtWidgets.QWidget(self.btn_frame)
        self.widget.setGeometry(QtCore.QRect(20, 10, 800, 90))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 20, 20, 20)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # 加载待检测图片
        self.btn_img_add_file = QtWidgets.QPushButton(self.widget)
        self.btn_img_add_file.setObjectName("btn_img_add_file")
        self.horizontalLayout.addWidget(self.btn_img_add_file)
        # # 选择图片
        # self.btn_model_add_file = QtWidgets.QPushButton(self.widget)
        # self.btn_model_add_file.setObjectName("btn_model_add_file")
        # self.horizontalLayout.addWidget(self.btn_model_add_file)
        # 开始检测
        self.btn_detect = QtWidgets.QPushButton(self.widget)
        self.btn_detect.setObjectName("btn_detect")
        self.horizontalLayout.addWidget(self.btn_detect)
        # 退出
        self.btn_exit = QtWidgets.QPushButton(self.widget)
        self.btn_exit.setObjectName("btn_exit")
        self.horizontalLayout.addWidget(self.btn_exit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1042, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        # 这里将按钮和定义的动作相连，通过click信号连接
        # 加载模型文件
        # self.btn_model_add_file.clicked.connect(self.open_model)
        # 显示待分割图片
        self.btn_img_add_file.clicked.connect(self.openimage)
        # 开始识别
        self.btn_detect.clicked.connect(self.object_detection)
        # 这里是将btn_exit按钮和Form窗口相连，点击按钮发送关闭窗口命令
        self.btn_exit.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        from PyQt5.QtCore import Qt
        self._translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(self._translate("MainWindow", "分割"))

        self.label_img_show.setText(self._translate("MainWindow", "待分割图片"))
        self.label_img_show.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        # self.info_show.setText(self._translate("MainWindow", "dice score"))
        # self.info_show.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.GT_show.setText(self._translate("MainWindow", "标准GT"))
        self.GT_show.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.label_detect_show.setText(self._translate("MainWindow", "预测结果"))
        self.label_detect_show.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.dice_show.setText(self._translate("MainWindow", "dice score:"))
        self.dice_show.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.btn_img_add_file.setText(self._translate("MainWindow", "选择图片"))
        # self.btn_model_add_file.setText(_translate("MainWindow", "加载模型文件"))
        self.btn_detect.setText(self._translate("MainWindow", "开始预测"))
        self.btn_exit.setText(self._translate("MainWindow", "退出"))
        self.openfile_name_mdoel=""
        self.sigmoid = torch.nn.Softmax()

    def open_model(self):
        self.openfile_name_mdoel, _ = QFileDialog.getOpenFileName(self.btn_model_add_file, '选择模型文件',
                                                             'D:\YZU-capstone\CA-Net\saved_models\ISIC2018')

        from Models.networks.network import Comprehensive_Atten_Unet
        args = get_parser()
        self.model = Comprehensive_Atten_Unet(args, args.num_input, args.num_classes).cuda()
        checkpoint = torch.load(self.openfile_name_mdoel)
        self.model.load_state_dict(checkpoint['state_dict'], False)
        self.model.eval()

        print('加载模型文件地址为：' + str(self.openfile_name_mdoel))

    def openimage(self):
        global fname
        # imgName = "D:/YZU-capstone/ISIC2018_Task1-2_Validation_Input/ISIC_0012255.jpg"
        imgName, imgType = QFileDialog.getOpenFileName(self.btn_img_add_file, "选择图片", "D:/YZU-capstone/ISIC2018_Task1-2_Validation_Input", "*.jpg;;*.png;;All Files(*)")
        self.img = cv2.imread(imgName, cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(self.img, ( self.img_width, self.img_height))
        # ISIC_0012255_segmentation.png ISIC_0012255.jpg
        gt_name = imgName.replace("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth")
        gt_name = gt_name.replace(".jpg", "_segmentation.png")
        self.gt = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
        self.gt = cv2.resize(self.gt, (self.img_width, self.img_height))
        # self.gt = self.gt/255.0
        showImage = cv2.cvtColor(self.gt, cv2.COLOR_GRAY2BGR)
        showGTImage = QtGui.QImage(showImage.tobytes(), showImage.shape[1], showImage.shape[0], showImage.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
        self.GT_show.setPixmap(QPixmap(showGTImage))
        # cv2.imshow("111", self.img)
        # cv2.waitkey(0)
        # self.img = self.img.transpose(2,0,1)
        jpg = QPixmap(imgName).scaled(self.img_width, self.img_height)
        self.label_img_show.setPixmap(jpg)

        fname = imgName

    def object_detection(self):
        print(fname)
        img = Image.open(fname)
        pred = self.predict_(img).cpu().numpy().astype('uint8').squeeze(0)
        ret, thresh = cv2.threshold(pred, 127, 255,0)
        con, hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.img, con, -1, (0,0,0), 2)
        dice_score = self.dice_loss(self.gt, pred)
        self.dice_show.setText(self._translate("MainWindow", "dice score:"+str(dice_score)))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], QtGui.QImage.Format_RGB888)
        self.label_detect_show.setPixmap(QPixmap(showImage))
        QApplication.processEvents()


    def predict_(self, img, model_file_path="D:\YZU-capstone\CA-Net\saved_models\ISIC2018\\folder2/min_loss_ISIC2018_checkpoint.pth.tar"):
        if self.openfile_name_mdoel == "":
            from Models.networks.network import Comprehensive_Atten_Unet
            args = get_parser()
            self.model = Comprehensive_Atten_Unet(args, args.num_input, args.num_classes).cuda()
            checkpoint = torch.load(model_file_path)
            self.model.load_state_dict(checkpoint['state_dict'], False)
            self.model.eval()

        print('开始预测')
        data_transform = transforms.Compose([
            transforms.Resize([224, 300]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        img = data_transform(img)
        img = img.float().cuda()
        img = img.unsqueeze(0)
        self.model = self.model.cuda()

        with torch.no_grad():
            output = self.model(img)

            output = output[:,1,:,:]  # b,224,300
            output = torch.nn.functional.interpolate(output.unsqueeze(1), (self.img_height, self.img_width),
                                                     mode='bilinear', align_corners=True)
            output = output.squeeze(1)
            output[output>=0.5] = 1
            output[output < 0.5] = 0
            output[output == 1] = 255
        print("-----------")
        return output

    def dice_loss(self, target, predictive, ep = 1e-8):
        predictive[predictive == 255] = 1
        target = target/255
        intersection = 2 * numpy.sum(predictive * target) + ep
        union = numpy.sum(predictive) + numpy.sum(target) + ep
        loss = intersection / union
        return loss



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    # 向主窗口上添加控件
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())