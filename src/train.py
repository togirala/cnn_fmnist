import dispatcher
import cnn1
import torch.nn.functional as F
import torch.optim as optim
import torch


def correct_preds(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()




def train():

    batch_loader = dispatcher.batch_loader
    images, labels = next(iter(batch_loader))

    model = cnn1.CNN1()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    
    for epoch in range(4):

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

    
    torch.save(model, f'{dispatcher.res_path}cnn1.pt')
    #return model






    



if __name__ == '__main__':
    train()






