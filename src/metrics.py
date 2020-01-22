from sklearn.metrics import confusion_matrix
import dispatcher
import torch



def get_all_preds(model, data_loader):
    
    all_preds = torch.tensor([])

    for batch in data_loader:
        images, labels = batch
        preds = model(images)

        all_preds = torch.cat((all_preds, preds), dim=0)
    
    return all_preds




def get_confusion_matrix(labels, predictions):
    
    cm = confusion_matrix(labels, predictions)
    
    return cm




def metrics():
    
    cnn1 = torch.load(f'{dispatcher.res_path}cnn1.pt')
    cnn1.eval()
    
    preds = get_all_preds(cnn1, dispatcher.batch_loader)
    cm = get_confusion_matrix(dispatcher.input_training_set.targets, preds.argmax(dim=1))






















if __name__ == '__main__':
    metrics()






















