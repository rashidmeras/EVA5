import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def initiate_training(model, device, train_loader, epochs=5):

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=1e-5, nesterov=False)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  
    criterion = nn.CrossEntropyLoss()
    
    ep = 1
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data[0].to(device) , data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                if (ep%6 == 0):
                    print('(*)[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                else:
                    print('(*)[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000), end=" ")
                running_loss = 0.0   
                ep += 1            
    
    print('Finished Training')
