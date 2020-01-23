from dispatcher import batch_loader, tb_status, res_path, parameters
import cnn1
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from itertools import product
from collections import OrderedDict
from collections import namedtuple


class Run():
    
    def __init__(self):
        
        self.params = None
        self.count = 0
        self.data = list()
        self.start_time = None

        self.tb = None

        self.data = None
        self.model = None


    def begin(self, run, model, data):

        self.start_time = time.time()
        
        self.params = run
        self.count += 1
        
        self.data = data
        self.model = model

        
        self.tb = SummmaryWriter(comment = f'-{run}')

        images, labels = next(iter(self.data))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.model, images)

        

    def end(self):

        
        self.Epoch.count = 0
        self.tb.close()



    def save(self, fileName):

        pd.DataFrame.from_dict(Run.data, orient='columns').to_csv(f'{fileNmae}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(Run.data, f, ensure_ascii=False, indent=4)






    

class Epoch(Run):
    
    def __init__(self):
        self.count = 0
        self.start_time = None
        self.loss = 0
        self.accuracy = 0
        

    def begin(self):

        self.start_time = time.time()
        self.count += 1
        self.loss = 0
        self.accuracy = 0


    def end(self):
        duration = time.time() - self.start_time
        
        loss = self.loss / len(Run.data.dataset)
        accuracy = self.accuracy / len(Run.data.dataset)

        self.tb.add_scalar('Loss: ', loss, self.count)
        self.tb.add_scalar('Accuracy: ', accuracy, self.count)

        for name, param in Run.model.named_parameters():
            self.add_histogram(name, param, self.count)
            self.add_histogram(f'{name}.grad', param.grad, self.count)

        results = OrderedDict()
        results['run'] = Run.count
        results['epoch'] = count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = duration
        results['run duration'] = Run.duration
        
        for k, v in Run.params._asdict().items(): results[k] = v

        Run.data.append(results)
        df = pd.DataFrame.from_dict(Run.data, orient = 'columns')

        
    def track_loss(self, loss):
        self.loss += loss.item() * Run.data.batch_size

    def track_accuracy(self, preds, labels):
        self.accuracy += self._get_accuracy(preds, labels)

    @torch.no_grad()
    def _get_accuracy(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()/len(Run.data.dataset)

    def save(self, fileName):

        pd.DataFrame.from_dict(Run.data, orient='columns').to_csv(f'{fileNmae}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(Run.data, f, ensure_ascii=False, indent=4)





# def correct_preds(preds, labels):
#     return preds.argmax(dim=1).eq(labels).sum().item()




def train():

    # param_values = [v for v in parameters.values()]
    
    r = Run()
    e = Epoch()
    for run in Dispatcher.get_runs():
        


        model = dispatcher.get_model()
        optimizer = optim.Adam(model.parameters(), lr=run.lr)
        batch_loader = Dispatcher.get_data_batch(batch_size = run.bs)
        
        r.begin(run, model, batch_loader)
        for epoch in range(5):
            
            e.begin()
            for batch in batch_loader:
                
                images = batch[0]
                labels = batch[1]

                preds = model(images) # forward prop ==> get predictions
                loss = F.cross_entropy(preds, labels) # calculate loss
                optimizer.zero_grad() # zero the grads
                loss.backward() # backward prop ==> calculate gradients
                optimizer.step() # update weights
        
                e.track_loss(loss)
                e.track_accuracy(preds, labels)

                e.end()

            r.end()

        r.save('results')
        print(f'run')

    
    torch.save(model, f'{res_path}cnn1.pt')
    #return model






    












if __name__ == '__main__':
    train()






