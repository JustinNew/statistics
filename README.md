Statistics
==========

### Area for normal distribution of s.d.

  - +/- s.d. 68%
  - +/- 2*s.d. 95%
  - +/- 3*s.d. 99.7%
  
### ROC

ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

  -  Y - TPR, X - FPR
  - TPR = TP / (TP + FN) = TP / P
  - FPR = FP / (FP + TN) = FP / N

Note:
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - Recall = TPR
  
Binary Classification Table

|      | Predicted Positive | Predicted Negative |
| ---- | ------------------ | ------------------ |
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

### K-S Test

https://onlinecourses.science.psu.edu/stat414/node/323

In statistics, the Kolmogorov–Smirnov test (K–S test or KS test) is a nonparametric test of the equality of continuous, one-dimensional probability distributions that can be used to compare a sample with a reference probability distribution (one-sample K–S test), or to compare two samples (two-sample K–S test).

### Simpson’s Paradox 

Simpson’s Paradox is a paradox in probability and statistics, in which a trend appears in different groups of data but disappears or reverses when these groups are combined.

### Gaussian Noise and Regularization

By adding Gaussian noise to the input, the learning model will behave like an L2-penalty regularizer. By adding Laplace noise to the input, the learning model will behave like an L1-penalty regularizer.

### Assumptions of linear regression

  - Linear relationship between Y and X1, …, Xp
  - Error Term i.i.d (Independent and identically distributed) 均值为0，近normal 分布(normally distributed residuals)，constant variance和independently distributed. 
  - The predictor variables X1, . . . , Xp, are assumed to be linearly independent of each other; variables are nonrandom; variable values are all measured without error.
  - All observations are equally reliable and have approximately equal role in determining the regression results and in influencing conclusions.

