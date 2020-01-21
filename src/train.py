import dispatcher
import cnn1
import torch.nn.functional as F
import torch.optim as optim



def correct_preds(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()




def train():

    data_loader = dispatcher.data_loader
    images, labels = next(iter(data_loader))

    model = cnn1.CNN1()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    '''
    ############## step by step iterations #######

    preds = model(images)
    loss = F.cross_entropy(preds, labels)
    print(f"initial loss: {loss.item()}")

    
    loss.backward() # Calculate Gradients
    optimizer.step() # Update weights

    preds = model(images)
    loss = F.cross_entropy(preds, labels)
    print(f"loss after one iter: {loss.item()}")
    '''
    

    total_loss = 0
    total_correct = 0

    for batch in data_loader:
        images, labels = batch
        
        preds = model(images) # forward prop ==> get predictions
        loss = F.cross_entropy(preds, labels) # calculate loss
        optimizer.zero_grad() # zero the grads
        loss.backward() # backward prop ==> calculate gradients
        optimizer.step() # update weights
        
        total_loss += loss.item()
        total_correct += correct_preds(preds, labels)

    print(f"total correct: {total_correct} and total loss: {total_loss}")









    



if __name__ == '__main__':
    train()






