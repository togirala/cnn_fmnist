import dispatcher
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from itertools import product
from collections import OrderedDict
from collections import namedtuple
import time
import pandas as pd
import json

class Run():
    
    def __init__(self):
        
        self.count = 0
        self.results = list()
        self.start_time = None
        self.duration = None
        self.tb = None

        self.data = None
        # self.model = model


    @staticmethod
    def get_runs(params):

        run = namedtuple('run', params.keys())
        runs = list()

        for v in product(*params.values()):
            runs.append(run(*v))

        return runs
    
    
    
    
    def begin(self, run, model, batch_loader):

        self.start_time = time.time()
        
        self.params = run
        self.count += 1
        
        # self.model = model
        # print(type(data))
        self.tb = SummaryWriter(comment = f'-{run}')
        
        images, labels = next(iter(batch_loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(model, images)

        

    def end(self, e):

        e.count = 0
        self.duration = time.time() - self.start_time
        self.tb.close()



    def save(self, fileName):

        pd.DataFrame.from_dict(self.results, orient='columns').to_csv(f'{fileName}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)






    

class Epoch():
    
    def __init__(self):
        self.count = 0
        self.start_time = None
        self.loss = None
        self.accuracy = None
        self.duration = None

    def begin(self):

        self.start_time = time.time()
        self.count += 1
        self.loss = 0
        self.accuracy = 0



    def end(self, run, model, data):
        self.duration = time.time() - self.start_time
        
        loss = self.loss / len(data.dataset)
        accuracy = self.accuracy / len(data.dataset)

        run.tb.add_scalar('Loss: ', loss, self.count)
        run.tb.add_scalar('Accuracy: ', accuracy, self.count)

        for name, param in model.named_parameters():
            run.tb.add_histogram(name, param, self.count)
            run.tb.add_histogram(f'{name}.grad', param.grad, self.count)

        results = OrderedDict()
        results['run'] = run.count
        results['epoch'] = self.count
        results['loss'] = self.loss
        results['accuracy'] = self.accuracy
        results['epoch duration'] = self.duration
        results['run duration'] = run.duration
        
        for k, v in run.params._asdict().items(): results[k] = v

        run.results.append(results)
        df = pd.DataFrame.from_dict(run.results, orient = 'columns')

        
    def track_loss(self, loss, batch_size):
        self.loss += loss.item() * batch_size

    def track_accuracy(self, preds, labels):
        self.accuracy += self._get_accuracy(preds, labels)


    # Current implementation; accuracy ==> no:of current predictions
    @torch.no_grad()
    def _get_accuracy(self, preds, labels):

        return preds.argmax(dim=1).eq(labels).sum().item()

    # currently not being used
    def save(self, fileName):

        pd.DataFrame.from_dict(run.results, orient='columns').to_csv(f'{fileNmae}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(run.results, f, ensure_ascii=False, indent=4)





# def correct_preds(preds, labels):
#     return preds.argmax(dim=1).eq(labels).sum().item()




def train():

    # param_values = [v for v in parameters.values()]
    d = dispatcher.Dispatcher()
    model = d.get_model()
    r = Run()
    e = Epoch()  ### continue from here
    for run in r.get_runs(d.get_parameters()):
        


        model = d.get_model()
        optimizer = optim.Adam(model.parameters(), lr=run.lr)
        batch_loader = d.get_data_batch(batch_size = run.bs)
    
        r.begin(run, model, batch_loader)
        for epoch in range(2):
            
            e.begin()
            for batch in batch_loader:
                
                images = batch[0]
                labels = batch[1]

                preds = model(images) # forward prop ==> get predictions
                loss = F.cross_entropy(preds, labels) # calculate loss
                optimizer.zero_grad() # zero the grads
                loss.backward() # backward prop ==> calculate gradients
                optimizer.step() # update weights
        
                e.track_loss(loss, run.bs)
                e.track_accuracy(preds, labels)

            # print(e.accuracy)
            e.end(run = r, model = model, data = batch_loader) ### continue from here


        print(r.results)
        r.end(e)
    
    res_path = d.get_resource_path()
    r.save(fileName = f'{res_path}results')
    torch.save(model, f'{res_path}cnn1.pt')

    #print(r.results)
    #return model






    












if __name__ == '__main__':
    train()






