import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch.utils.data 
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import cnn1



class Dispatcher():


    def __init__(self):
        self.res_path = None
        self.training_set = None
        self.test_set = None
        self.parameters = None
        self.model = None
    
    def get_model(self):

        return cnn1.CNN1()


    def get_parameters(self):
        
        parameters = OrderedDict(
                lr = [0.01, 0.001]
                ,bs = [100, 1000]
                # ,shuffle = []
                )

        return parameters
        

    def get_dataset(self):
        self.training_set = ds.FashinMNIST(
                root = 'input/',
                train = True,
                download = True,
                transform = transforms.Compose([transforms.ToTensor()])
                )



    def get_data_batch(self, batch_size = 256, shuffle = False, num_workers = 1):
        batch_data = torch.utils.data.DataLoader(
                dataset = get_dataset()
                batch_size = batch_size,
                shuffle = shuffle
                num_workers = num_workers
                )
    
        return batch_data

    
    @staticmethod
    def get_runs(self):
        
        params = get_parameters()
        run = namedtuple('run', param.keys())
        runs = list()
 
        for v in product(*params.values()):
            runs.append(run(*v))

        return runs






    #res_path = 'models/'



            













