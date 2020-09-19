* Objective: Take given code and meet the following targets:

1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (basically 3 MPs)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. achieve 80% accuracy, as many epochs as you want. 
8. Total Params to be less than 1M. 

Result:
A model was created meeting the given targets with and trained for 25 epochs
* Total params: 926,794
* Accuracy of the network on the 10000 test images: 81 %

Submitted By: 
* `Meras Pillai Rashid`
* `Samir Prasad`
