import dispatcher
import torch.nn as nn
import torch.nn.functional as F



class CNN2(nn.Module):
    
    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 12, kernel_size=3)
    

        self.fc1 = nn.Linear(in_features = 12*16*16, out_features = 200)
        self.fc2 = nn.Linear(in_features = 200, out_features = 80)
        self.out = nn.Linear(in_features = 80, out_features = 10)


    def forward(self, t):
        # 1st Layer
        t = t
        # print(f"shape of tensor at input layer is {t.shape}")
        
        
        # 2nd layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 3, stride = 1)
        # print(f"shape of tensor after 2nd layer is {t.shape}")


        # 3rd layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 3, stride = 1)
        # print(f"shape of tensor after 3rd layer is {t.shape}")



        # 4th layer
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 3, stride = 1)
        # print(f'shape of tensor after 4th layer is {t.shape}')


        # 5th Layer
        t = t.reshape(-1, 12*16*16)
        t = self.fc1(t)
        t = F.relu(t)
        # print(f"shape of tensor after 4th layer is {t.shape}")


        # 5th Layer
        t = self.fc2(t)
        t = F.relu(t)
        # print(f"shape of tensor after 5th layer is {t.shape}")


        # 7th layer
        t = self.out(t)
        # t = F.softmax(t, dims = 1)
        # print(f"shape of tensor after 6th layer is {t.shape}")

        
        
        

        
        return t





















