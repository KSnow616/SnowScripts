from matplotlib import pyplot as plt
import torch, torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training script for testing pytorch & cuda
# Requires numpy

cuda_enabled = torch.cuda.is_available()
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])])

data_train = datasets.MNIST(root = "./data/",transform=transform,train=True,download=True)
data_test = datasets.MNIST(root = "./data/",transform=transform,train=False)


data_loader_train = torch.utils.data.DataLoader(dataset=data_train,batch_size=64,shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,batch_size=64,shuffle=True)

# show images
"""
images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)
plt.show()
"""

class Model(torch.nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.conv1=torch.nn.Sequential(
                    torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense=torch.nn.Sequential(
                    torch.nn.Linear(14*14*128,1024),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(1024,10))
    def forward(self,x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)
        x=self.dense(x)
        return x

model=Model()
cost=torch.nn.CrossEntropyLoss()
optimzer=torch.optim.Adam(model.parameters())
# print(model)

if(cuda_enabled):
    model.cuda()

n_epoch=1
for epoch in range(n_epoch):
    running_loss=0.0
    running_correct=0
    print("Epoch{}/{}".format(epoch,n_epoch)) 
    print("-"*10)
    for data in data_loader_train: 
        X_train,Y_train=data
        if(cuda_enabled):
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
        X_train,Y_train=Variable(X_train),Variable(Y_train)
        outputs=model(X_train)
        _,pred=torch.max(outputs.data,1)
        if(cuda_enabled):
            pred = pred.cuda()
        optimzer.zero_grad()
        loss=cost(outputs,Y_train)
        
        loss.backward() 
        optimzer.step()
        running_loss+=loss.item()
        running_correct+=torch.sum(pred==Y_train.data)
    testing_correct=0
    for data in data_loader_test:
        X_test,Y_test=data
        if(cuda_enabled):
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()

        X_test,Y_test=Variable(X_test),Variable(Y_test)
        outputs=model(X_test)
        _,pred=torch.max(outputs.data,1)
        if(cuda_enabled):
            pred = pred.cuda()
        testing_correct+=torch.sum(pred==Y_test.data)
    print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                 100*running_correct/len(data_train),100*testing_correct/len(data_test)))

# test model
data_loader_test=torch.utils.data.DataLoader(dataset=data_test,
                                            batch_size=64,
                                            shuffle=True)

X_test,Y_test=next(iter(data_loader_test))
if(cuda_enabled):
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()
inputs=Variable(X_test)
pred=model(inputs)
_,pred=torch.max(pred,1)
if(cuda_enabled):
    pred = pred.cpu()

print("Predict Label is:",[i for i in pred.data])
print("real    Label is:",[i for i in Y_test])

if(cuda_enabled):
    X_test = X_test.cpu()

img=torchvision.utils.make_grid(X_test)
img=img.numpy().transpose(1,2,0)

std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
plt.imshow(img)
plt.show()
