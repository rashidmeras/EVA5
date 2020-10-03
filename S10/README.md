Objective:

* Pick your last code and Add CutOut. It should come from your transformations (albumentations)
* Use this repo: https://github.com/davidtvs/pytorch-lr-finder
* Move LR Finder code to your modules
* Implement LR Finder (for SGD, not for ADAM)
* Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
* Find best LR to train your model
* Use SDG with Momentum
* Train for 50 Epochs. 
* Show Training and Test Accuracy curves
* Target 88% Accuracy.
* Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.

Summary:
* A ResNet18 model was used and Albumination functions were imported to implement transformations including Cutout. LR finder was implemented and best LR was found and the network was trained for 50 epochs.

* Total params: `11,173,962`
* Accuracy of the network on the 10000 test images:  `85%`

Submitted By: 
* `Meras Pillai Rashid`
* `Samir Prasad`
