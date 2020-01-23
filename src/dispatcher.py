import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch.utils.data 
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import cnn1
import torch


class Dispatcher():


    def __init__(self):
        self.res_path = None
        self.training_set = None
        self.test_set = None
        self.parameters = None
        self.model = None
        self.batch_data = None
        self.resource_path = None
        self.device = None    


    def get_model(self):
        
        self.model = cnn1.CNN1().to(get_device())
        
        return self.model



    def get_parameters(self):
        
        self.parameters = OrderedDict(
                lr = [0.01, 0.001]
                ,bs = [100, 1000]
                # ,shuffle = []
                )

        return self.parameters
        


    def get_dataset(self):
        return ds.FashionMNIST(
                root = 'input/',
                train = True,
                download = True,
                transform = transforms.Compose([transforms.ToTensor()])
                )

        

    def get_data_batch(self, batch_size = 256, shuffle = False, num_workers = 1):
        
        self.batch_data = torch.utils.data.DataLoader(
                dataset = self.get_dataset(),
                batch_size = batch_size,
                shuffle = shuffle,
                num_workers = num_workers
                )
    
        return self.batch_data



    def get_resource_path(self):

        self.resource_path = 'models/'
        
        return self.resource_path
    

    def get_device(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        return self.device










