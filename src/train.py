import dispatcher
import cnn1




def train():

    data_loader = dispatcher.data_loader

    image, label = next(iter(data_loader))


    print(image.shape)
    print(label)


    model = cnn1.CNN1()

    output = model(image)

    # print(output)




if __name__ == '__main__':
    train()






