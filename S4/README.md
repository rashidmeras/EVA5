## Architectural Basics

Here the objective is to create a Vanilla network for MNIST data set and define the network architecture such that:
1. It has 3x3 convolution
2. It has 1x1 convolution
3. It has 7x7 convolution
4. It has a transitional layer (i.e. MaxPooling)
5. Use ReLU for activation function
6. Learning Rate, Batch Normalization, DropOut
7. Use SoftMax at the final layer

The main objective at this level is to define the architecture of the network and then work towards tuning the network using different techniques. 

The proposed network Architecture:

![Proposed Network Archtecture](https://rashidmeras.github.io/images/eva/S4_Proposal3_Fig2.png)

The above figure shows the propsed network architecture. As shown in the figure there are a total of 11 layers in the network. Starting from 32 channels at Layer1 the cahnnel size is doubled at each layer and at Layer9 the channles size increses maximum upto 1024 channels.

Using this architecture we derive a Network which is explored in detail.

The result of this experiment is as shown below:
* Total params: 19,618
* Epochs = 10
* Accuracy = `99.45%`
