import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch.utils.data 



class Dispatcher():


    def __init__(self):
        self.res_path = None
        self.training_set = None
        self.test_set = None
        self.parameters = None


    def get_parameters():
        
        parameters = dict(
                lr = []
                ,bs = []
                ,shuffle = []
                )


        return parameters
        

    def get_data_batch(data_set):
        batch_data = torch.utils.data.DataLoader(
                data_set
                ,batch_size = 1000
                ,shuffle = False
                ,num_workers = 1
                )
    
        return batch_data


    





    res_path = 'models/'


    input_training_set = ds.FashionMNIST(
        root='input/', 
        train = True, 
        # transform=None, 
        target_transform=None, 
        download=True,
        transform = transforms.Compose([transforms.ToTensor()])
        )

        # data_loader = data.DataLoader(input_training_set, batch_size=4, shuffle=True, num_workers=args.nThreads)
    batch_loader = data.DataLoader(input_training_set, batch_size=1000, shuffle=False, num_workers=1)


    tb_status = True


    parameters = dict(
            lr = [0.01, 0.001],
            bs = [10, 100, 1000]#,
            #shuffle = [True, False]
            )
            













