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
  
### With or Without Replacement

With Replacement
  - Sampling with replacement is used to find probability with replacement. In other words, you want to find the probability of some event where there’s a number of balls, cards or other objects, and you replace the item each time you choose one. Let’s say you had a population of 7 people, and you wanted to sample 2. Their names are: John, Jack, Qiu, Tina, Hatty, Jacques, Des. You could put their names in a hat. If you sample with replacement, you would choose one person’s name, put that person’s name back in the hat, and then choose another name. The possibilities for your two-name sample are: [John, John], [John, Jack], [John, Qui], … and so on.

Without Replacement
  - Sampling without Replacement is a way to figure out probability without replacement. In other words, you don’t replace the first item you choose before you choose a second.

# Problems

### A box contains a red ball, an orange ball, a yellow ball, a green ball, a blue ball and an indigo ball. Billy randomly selects 7 balls from the box (with replacement). What is the expected value for the number of distinct colored balls Billy will select?
  - Let P(i),0<i<7 be the probability of having i distinct balls after drawing 7 balls.
  - P(1) = C(6,1) * (1/6)^7
  - P(2) = C(6,2) * (1/6)^7 * (C(7,1) + C(7,2) + C(7,3) + C(7,4) + C(7,5) + C(7,6))
  - ...
  
### A fair cube with 6 faces. On average, how many times do you need to roll the cube to get #1?

  - roll once and get #1: 1/6
  - roll twice and get #1: 5/6*1/6
  - roll 3 times and get #1: (5/6)^2*1/6
  - ...
  - E(N) = 1/6 + 2*1/6*5/6 + 3*1/6*(5/6)^2 + ...
  - 5/6 * E(N) = 1/6 * 5/6 + ... 
  - 1/6 * E(N) = 1/6 + 1/6*5/6 + = 1
  - E(N) = 1/p

#### How many times to roll to get a set (1,2,3,4,5,6)?

  - 1th: a, 1
  - 2nd: b, 5/6
  - 3rd: c, 4/6
  - 4th: d, 3/6
  - 5th: e, 2/6
  - 6th: f, 1/6
  - To get (1,2,3,4,5,6): 1 + 6/5 + 6/4 + 6/3 + 6/2 + 6/1

#### 1000 people, each time select 10 (w/o replacement) 问每个人on average多少次会被抽到

  - In sampling without replacement, focus on how many iterations are needed. That number will be any count from 1 through 100, each equally likely. 
  - So sampling without replacement, it will be 1/100*(1+2+...+100) = 50.5.

#### 1000 people, each time select 10, (w/ replacement), 问每个人on average多少次会被第一次抽到（1000个人，你每次抽10个人。 每次抽完以后再把这10个人放回去。这样有些人会被抽中1次，有些人会被抽中2次，etc。 比如你抽了10次之后，有一个人Mike, 他第一次被你抽中。现在问，平均下来，一个人被第一次抽中时， 你抽了几次。）

  - p = 1 / 100
  - Expected times to be selected 1/p = 100
  
### If the probability a car get at least one ticket within an hour is 84%, what is the probability of getting at least one ticket in half an hour?

  - Say the probability of not get a ticket within half an hour is 1-x, then one hour without a ticket is (1-x)*(1-x)=1-0.84. x=0.6
  
or, 

  - It really depends on what model is assumed. However, if the idea is that no matter how long you leave your car there, you have a 16% chance of getting through any given hour unscathed, you can treat it as an exponential decay problem. Let p(t) be the probability that you do not get a ticket in the first t hours. Then p(1)=0.16, p(2)=0.16*0.16 (a 16% chance of making it through the first hour times a 16% chance of making it through the second), and in general p(t)=0.16^t. The probability of not getting a ticket in the first half hour is then p(1/2)=0.16^(1/2)=0.4, and the probability that you do get a ticket in the first half hour is about 1−0.4=0.6.
  
### I throw three (n) darts along a circle, what is the probability all three(n) points are in a semi-circle?

  - Suppose that point i has angle 0 (angle is arbitrary in this problem) -- essentially this is the event that point i is the "first" or "leading" point in the semicircle. Then we want the event that all of the points are in the same semicircle -- i.e., that the remaining points end up all in the upper half plane.

  - That's a coin-flip for each remaining point, so you end up with 1 / 2^(n−1). There's n points, and the event that any point i is the "leading" point is disjoint from the event that any other point j is, so the final probability is n / 2^(n−1) (i.e. we can just add them up).

  - A sanity check for this answer is to notice that if you have either one or two points, then the probability must be 1, which is true in both cases.
  
### 1) A and B have a game. There are 7 games in total, who wins 4 games first will succeed the whole game and then the game ends. Given that A has a probability P to win a single game, and A already lost the first 2 games, what's the probability that A still wins the whole game. 2) Given that someone said A has 80% probability to win the whole game, what is the posterior that A win the whole game.'

### Consider a game with 2 players, A and B. Player A has 8 stones, player B has 6. Game proceeds as follows. First, A rolls a fair 6-sided die, and the number on the die determines how many stones A takes over from B. Next, B rolls the same die, and the exact same thing happens in reverse. This concludes the round. Whoever has more stones at the end of the round wins and the game is over. If players end up with equal # of stones at the end of the round, it is a tie and another round ensues. What is the probability that B wins in 1, 2, ..., n rounds?

  - On the first round, B can win if (A,B) rolls: (1,4), (1,5), (1,6), (2,5), (2,6), (3,6) - so there are 6 out of 36 possibilities where B wins. P(B1) [read, probability that B wins in round 1] = 1/6 On the first round, B can tie if (A,B) rolls: (1,3), (2,4), (3,5), (4,6) - so there are 4 out of 36 possibilities where B ties. If B ties with A, there is a second round, where B wins with probability 15/36, A wins with probability 15/36 = 5/12, and they tie with probability 6/36. So P(B2) = 1/9 * (5/12) After the second round, the game only continues if both players have equal number of points, and the probability of a tie in each game is 1/6, so P(B3) = (1/9)*(1/6)*(5/12) Generally, P(Bn) = (1/9)*(1/6)^(n-2)*(5/12)





