from dispatcher import batch_loader, tb_status, res_path, parameters
import cnn1
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from itertools import product




def correct_preds(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()




def train():

    param_values = [v for v in parameters.values()]
    
    for lr, bs in product(*param_values):
        images, labels = next(iter(batch_loader))

        model = cnn1.CNN1()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    


        if tb_status:
        
            comment = f'Batch size={bs}  Learning rate={lr}'
            tb = SummaryWriter(comment=comment)

            grid = torchvision.utils.make_grid(images)
            tb.add_image('image', grid)
            tb.add_graph(model, images)

        
    
        for epoch in range(5):

            total_loss = 0
            total_correct = 0

            for batch in batch_loader:
                images, labels = batch
        
                preds = model(images) # forward prop ==> get predictions
                loss = F.cross_entropy(preds, labels) # calculate loss
                optimizer.zero_grad() # zero the grads
                loss.backward() # backward prop ==> calculate gradients
                optimizer.step() # update weights
        
                total_loss += loss.item() * bs
                total_correct += correct_preds(preds, labels)

            # print(f"epoch: {epoch} ==> total correct: {total_correct} and total loss: {total_loss}")
        
            if tb_status:
                tb.add_scalar('Loss', total_loss, epoch)
                tb.add_scalar('Total correct', total_correct, epoch)

                for name, weight in model.named_parameters():
                    tb.add_histogram(name, weight, epoch)
                    tb.add_histogram(f'{name}.grad', weight.grad, epoch)


        print(f"Learning Rate: {lr} and Batch Size: {bs} ==> total correct: {total_correct} and total loss: {total_loss}")
        
        
        if tb_status:
            tb.close()


    
    torch.save(model, f'{res_path}cnn1.pt')
    #return model






    












if __name__ == '__main__':
    train()






