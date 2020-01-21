import dispatcher
import cnn1
import torch.nn.functional as F
import torch.optim as optim



def train():

    data_loader = dispatcher.data_loader
    images, labels = next(iter(data_loader))

    model = cnn1.CNN1()
    
    preds = model(images)
    loss = F.cross_entropy(preds, labels)
    print(f"initial loss: {loss.item()}")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss.backward() # Calculate Gradients
    optimizer.step() # Update weights

    preds = model(images)
    loss = F.cross_entropy(preds, labels)
    print(f"loss after one iter: {loss.item()}")

    
    

    # print(output)




if __name__ == '__main__':
    train()






