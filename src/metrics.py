from sklearn.metrics import confusion_matrix
import dispatcher
import torch
# import train


def get_all_preds(model, data_loader):
    
    all_preds = torch.tensor([])

    for batch in data_loader:
        images, labels = batch
        preds = model(images)

        all_preds = torch.cat((all_preds, preds), dim=0)
    
    # print(all_preds.shape)
    return all_preds




def get_confusion_matrix(labels, predictions):
    

    cm = confusion_matrix(labels, predictions)
    return cm



def metrics():
    
    cnn1 = torch.load(f'{dispatcher.res_path}cnn1.pt')
    cnn1.eval()
    #print(cnn1)
    
    #cnn1 = train.train()
    preds = get_all_preds(cnn1, dispatcher.batch_loader)
    
    print(dispatcher.input_training_set.targets[:10])
    print(preds.argmax(dim=1)[:10])
    cm = get_confusion_matrix(dispatcher.input_training_set.targets, preds.argmax(dim=1))
    print(cm)


    #print(dispatcher.input_training_set.targets)
    


    #print(dispatcher.input_training_set.shape)
    #print(dispatcher.input_training_set.target)



# cm = confusion_matrix(dispatcher.input_training_set.target)




if __name__ == '__main__':
    metrics()






















