from model.AlexNet import QAlexNet
from model.VGGNet import QVGGNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

# parameter
num_classes = 2
num_epoch = 50
batch_size = 64
learning_rate = 0.001
image_size = 227

train_data_path = '../dataset/testset/train'
test_data_path = '../dataset/testset/test'
save_path = '../backup/new_model.pth'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])


# data loading
train_datasets = datasets.ImageFolder(root=train_data_path, transform=transform)
train_dataloader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)

test_datasets = datasets.ImageFolder(root=test_data_path, transform=transform)
test_dataloader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)


# load model
model = QAlexNet(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()


# # training
# for epoch in range(num_epoch):
#     model.train()
#     loss_arr = []
#
#     for batch, data in enumerate(train_dataloader):
#         x, label = data
#         if torch.cuda.is_available():
#             x = x.cuda()
#             label = label.cuda()
#
#         output = model(x)
#
#         optim.zero_grad()
#         loss = loss_func(output, label)
#
#         loss.backward()
#         optim.step()
#
#         print('epoch [{}/{}] | batch [{}/{}] | loss : {:.4f}'.format(
#             epoch, num_epoch, batch, len(train_dataloader), loss.item()))
#
#         loss_arr.append(loss.item())
#
#     print('epoch [{}/{}] | loss mean : {:.4f}'.format(
#         epoch, num_epoch, np.mean(loss_arr)))
#
#     model.eval()
#     with torch.no_grad():
#         val_loss_arr = []
#         for batch, data in enumerate(test_dataloader):
#             x, label = data
#             if torch.cuda.is_available():
#                 x = x.cuda()
#                 label = label.cuda()
#
#             output = model(x)
#             val_loss = loss_func(output, label)
#             val_loss_arr.append(val_loss.item())
#
#         print('epoch [{}/{}] | val loss mean : {:.4f}'.format(
#             epoch, num_epoch, np.mean(val_loss_arr)))
#
# # model save
# torch.save(model, '../backup/new_model.pth')

# load save model
load_model = torch.load('../backup/new_model.pth')
load_model.eval()

# cal accuracy
total_num = 0
correct_num = 0

for data, label in test_dataloader:
    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()

    output = load_model(data)
    _, predict = torch.max(output.data, 1)

    total_num += label.size(0)
    correct_num += (predict == label).sum().item()

print("Accuracy of Test Data : {:.2f} %".format(100 * correct_num / total_num))
