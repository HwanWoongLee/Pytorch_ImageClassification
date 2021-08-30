## Pytorch_ImageClassification
- Pytorch
- Classification
- AlexNet, VGGNet, ResNet

## Version
- torch : 1.8.0+cu111
- torchvision : 0.9.0+cu111
- python : 3.7.6
- numpy : 1.18.2

## Usage Exaple
### 1. Set Parameter
```
# parameter
num_classes = 2
num_epoch = 50
batch_size = 64
learning_rate = 0.001
image_size = 224

train_data_path = '../dataset/testset/train'
test_data_path = '../dataset/testset/test'
save_path = '../backup/new_model.pth'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
```
- num_classes : number of class
- num_epoch : train epoch
- batch_size : train dataset batch size
- image_size : check model input size

### 2. Select model
```
# load model
model = QResNet152(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```
- QAlexNet (image size : 227x227)
- QVGGNet (224x224)
- QResNet34, QResNet152 (224x224)

### 3. Training & Test

![train_graph](https://user-images.githubusercontent.com/20108771/131287567-fcf0fafd-d2b8-4e14-8ab1-8feb72089692.png)


