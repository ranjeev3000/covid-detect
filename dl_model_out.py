#Importing the necessary packages:

import os
import PIL
import sys
import torch
import torchvision
import numpy as np 
import pandas as pd
import torch.nn as nn
from time import time
from PIL import Image
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
from matplotlib.pyplot import imshow
import torchvision.transforms as transforms
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
# lime import
from lime import lime_image
from skimage.segmentation import mark_boundaries

BASE_PATH=r'C:/Users/Admin/AppData/Local/Programs/Flask_Thesis/data1/'

train_dataset=pd.read_csv(os.path.join(BASE_PATH,'train_labels.csv'))
test_dataset=pd.read_csv(os.path.join(BASE_PATH,'test_labels.csv'))

#print(train_dataset.head(3))
#print(test_dataset.head(3))

#Plot Images to take a look at the X-Ray images
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(32, 32)) #Size of Figure
columns = 3 #Columns in fig
rows = 5 #Rows in Fig
for i in range(1,rows*columns+1):
    IMG_PATH=BASE_PATH+'train/'
    img=Image.open(os.path.join(IMG_PATH,train_dataset.iloc[i][0]))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

#Define the Dataset class to pair images and labels

class Dataset(data.Dataset):
    def __init__(self,csv_path,images_path,transform=None):
        self.train_set=pd.read_csv(csv_path) #Read The CSV and create the dataframe
        self.train_path=images_path #Images Path
        self.transform=transform # Augmentation Transforms
    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self,idx):
        file_name=self.train_set.iloc[idx][0] 
        label=self.train_set.iloc[idx][1]
        img=Image.open(os.path.join(self.train_path,file_name)) #Loading Image
        if self.transform is not None:
            img=self.transform(img)
        return img,label
    
learning_rate=1e-4
training_set_untransformed=Dataset(os.path.join(BASE_PATH,'train_labels.csv'),os.path.join(BASE_PATH,'train/'))
print(type(training_set_untransformed))

# Define a transform operation that applies transformations to an image

transform_train = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
		transforms.ToTensor()])
    
#Create transformed images from the training dataset such that the minority class is upsampled and the 
#resulting classes are equal in number in the new set

new_created_images=[]
for j in range (len(training_set_untransformed)):
    if training_set_untransformed[j][1]==1:
        for k in range(8):
            transformed_image = transform_train(training_set_untransformed[j][0])
            new_created_images.append((transformed_image,1))
    else:
        transformed_image = transform_train(training_set_untransformed[j][0])
        new_created_images.append((transformed_image,0))

print(len(new_created_images))   

# Split the new set into a training and validation dataset in the 80:20 ratio

train_size = int(0.8 * len(new_created_images))
validation_size = len(new_created_images) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(new_created_images, [train_size,validation_size])

training_generator = data.DataLoader(train_dataset,shuffle=True,batch_size=32,pin_memory=True) 

# Enable GPU computation

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# Instantiate Efficientnet3 
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
model.to(device)

# Display the summary of the model

#print(summary(model, input_size=(3, 224, 224)))

# Create a folder in the stat946winter2021 directory to save Weights

PATH_SAVE='./Weights/'
if(not os.path.exists(PATH_SAVE)):
    os.mkdir(PATH_SAVE)

# Make crossentropyloss as the criterion, set a learning rate decay and use Adam or weight update

criterion = nn.CrossEntropyLoss()
lr_decay=0.99
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#Create a class list

eye = torch.eye(2).to(device)
classes=[0,1]

# Create lists to record accuracy and loss and set the number of epochs ( I got 11 as the optimal number of epochs)

history_accuracy=[]
history_loss=[]
epochs = 1

# Train the model

for epoch in range(epochs):  
    running_loss = 0.0
    correct=0
    total=0
    class_correct = list(0. for _ in classes)
    class_total = list(0. for _ in classes)
    
    for i, data in enumerate(training_generator, 0):
        inputs, labels = data
        t0 = time()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = eye[labels]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        c = (predicted == labels.data).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = float(correct) / float(total)
        
        history_accuracy.append(accuracy)
        history_loss.append(loss)
        
        loss.backward()
        optimizer.step()
        
        for j in range(labels.size(0)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1
        
        running_loss += loss.item()
        
        print( "Epoch : ",epoch+1," Batch : ", i+1," Loss :  ",running_loss/(i+1)," Accuracy : ",accuracy,"Time ",round(time()-t0, 2),"s" )
    for k in range(len(classes)):
        if(class_total[k]!=0):
            print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
        
    print('[%d epoch] Accuracy of the network on the Training images: %d %%' % (epoch+1, 100 * correct / total))
    
    if epoch%10==0 or epoch==0:
        torch.save(model.state_dict(), os.path.join(PATH_SAVE,str(epoch+1)+'_'+str(accuracy)+'.pth'))
        
torch.save(model.state_dict(), os.path.join(PATH_SAVE,'Last_epoch'+str(accuracy)+'.pth'))

# Apply the same transformations on the test set as the training set so that the data distribution remains the same 

test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.RandomApply([
        torchvision.transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip()],0.7),
                                      transforms.ToTensor(),
                                     ])
    
# Create a data frame from the sample submission file

submission=pd.read_csv(BASE_PATH+'sample_submission.csv')

# LIME XAI:
import torch.nn.functional as F

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

'''

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=1000)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)

'''
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as imggg:
            return imggg.convert('RGB')

def get_explanation(image):
  imggg = get_image(image)
  explainer = lime_image.LimeImageExplainer()
  explanation = explainer.explain_instance(np.array(pill_transf(imggg)), 
                                         batch_predict, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=1000) #
  explanation.top_labels[0]
  return explanation


def get_lime_image1(explanation):
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
  img_boundry1 = mark_boundaries(temp/255.0, mask)
  plt.imshow(img_boundry1)
  plt.savefig('.\static\images\Lime_DL\image_positive.png')

def get_lime_image2(explanation):
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
  img_boundry2 = mark_boundaries(temp/255.0, mask)
  plt.imshow(img_boundry2)
  plt.savefig('.\static\images\Lime_DL\image_negative.png')