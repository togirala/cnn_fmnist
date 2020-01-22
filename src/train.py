import dispatcher
import cnn1
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision





def correct_preds(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()




def train():

    batch_loader = dispatcher.batch_loader
    images, labels = next(iter(batch_loader))

    model = cnn1.CNN1()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    if dispatcher.tb_status:
        tb = SummaryWriter()
        grid = torchvision.utils.make_grid(images)
        tb.add_image('image', grid)
        tb.add_graph(model, images)

    
    for epoch in range(10):

        total_loss = 0
        total_correct = 0

        for batch in batch_loader:
            images, labels = batch
        
            preds = model(images) # forward prop ==> get predictions
            loss = F.cross_entropy(preds, labels) # calculate loss
            optimizer.zero_grad() # zero the grads
            loss.backward() # backward prop ==> calculate gradients
            optimizer.step() # update weights
        
            total_loss += loss.item()
            total_correct += correct_preds(preds, labels)

        print(f"epoch: {epoch} ==> total correct: {total_correct} and total loss: {total_loss}")
        
        if dispatcher.tb_status:
            tb.add_scalar('Loss', total_loss, epoch)
            tb.add_scalar('Total correct', total_correct, epoch)

            tb.add_histogram('conv1.bias', model.conv1.bias, epoch)
            tb.add_histogram('conv1.weight', model.conv1.weight, epoch)
            tb.add_histogram('conv1.weight.grad', model.conv1.weight.grad, epoch)


    if dispatcher.tb_status:
        tb.close()


    
    torch.save(model, f'{dispatcher.res_path}cnn1.pt')
    #return model






    












if __name__ == '__main__':
    train()






