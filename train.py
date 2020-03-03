from dataloader.cifar10 import cifar
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from models.preceptron import preceptron
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def train(net, train_iter, device, loss, num_epochs, optimizer, debug_steps=200):
    net.train()
    save_epoch =10
    for epoch in range(num_epochs):
        print("Begin train epoch {}".format(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_iter):
            image, target = data
            image = image.to(device)
            target = target.to(device)
            y = net(image)
            optimizer.zero_grad()
            losses = loss(y, target)
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if i+1 % debug_steps == 0:
                avg_loss = running_loss / debug_steps
                logging.info(
                    f"Epoch: {epoch}, Step: {i}, " +
                    f"Average Loss: {avg_loss:.4f}, "
                )
                running_loss = 0.0
        if (epoch+1) % save_epoch == 0:
            #save model
            state = {'net':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch
            }
            save_path = './output/epoch-{}-loss-{}.pth'.format(epoch, avg_loss)
            torch.save(state, save_path)

def test(net, loader, loss,  device):
    net.eval()
    running_loss = 0.0
    num = len(loader)
    acc_sum = 0

    pbar = tqdm(total=num)
    for _, data in enumerate(loader):
        image, target = data
        image = image.to(device)
        target = target.to(device)
        with torch.no_grad():
            y = net(image)
            losses = loss(y, target)
            acc_sum += (y.argmax(dim=1) == target).sum().item()
        running_loss += losses.item()
        pbar.update(1)
    pbar.close()

    return running_loss/num, acc_sum/num


if __name__ == '__main__':
    dataset = cifar()
    trainloader = DataLoader(dataset.cifar_train, batch_size=32,
                             shuffle=True, num_workers=0)
    testloader = DataLoader(dataset.cifar_test, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = preceptron(feature_size=dataset.pca_channel, hidden_size=1280, num_classes=len(dataset.classes),device=device)

    optimizer = torch.optim.SGD(net.params, lr=0.001, momentum=0.9,
                                weight_decay=5e-4)



    loss = torch.nn.CrossEntropyLoss()
    train(net, trainloader, device, loss, num_epochs=100, optimizer=optimizer)
    losses, acc = test(net, testloader, loss, device)
    print('Test accuracy is {:.4f}%'.format(acc))
    print('Test loss is {:.4f}.'.format(losses))