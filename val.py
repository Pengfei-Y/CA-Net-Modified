from opts import get_parser
import  torchvision.transforms as transforms
import torch
from PIL import Image
from einops import rearrange

def predict_(img_name, model_file_path="D:\YZU-capstone\CA-Net\saved_models\ISIC2018\\folder2/min_loss_ISIC2018_checkpoint.pth.tar"):
    img = Image.open(img_name)

    from Models.networks.network import Comprehensive_Atten_Unet
    args = get_parser()
    model = Comprehensive_Atten_Unet(args, args.num_input, args.num_classes).cuda()
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print('开始预测')
    data_transform = transforms.Compose([
        transforms.Resize([224, 300]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    img = data_transform(img)
    img = img.float().cuda()
    img = img.unsqueeze(0)
    print(img.shape)
    model = model.cuda()
    print("-----------")

    with torch.no_grad():
        output = model(img)
        output[output >= 0.5] = 1;
        output[output < 0.5] = 0;
        output[output == 1] = 255;
    return output


predict_(img_name="D:\YZU-capstone\ISIC2018_Task1-2_Test_Input\ISIC_0012236.jpg")