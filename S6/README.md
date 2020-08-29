The objective of this project is to take best 5th code from S5, and run below versions for 25 epochs and report findings:

1. with L1 + BN
2. with L2 + BN
3. with L1 and L2 with BN
4. with GBN
5. with L1 and L2 with GBN

* Draw ONE graph to show the validation accuracy curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.

* Draw ONE graph to show the loss change curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.

* Find any 25 misclassified images (combined into single image) for "with GBN" model. Use the saved model from the above jobs and show the actual and predicted class names.

The code is implemented in a highly modularized way and a single function was used to iterate through these conditions

Following are the accuracy observed:

1. with L1 + BN => **99.29%**
2. with L2 + BN => **99.31%**
3. with L1 and L2 with BN => **99.47%**
4. with GBN => **98.9%**
5. with L1 and L2 with GBN => **99.33%**
