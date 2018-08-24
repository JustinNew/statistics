Statistics
==========

### Area for normal distribution of s.d.

  - +/- s.d. 68%
  - +/- 2*s.d. 95%
  - +/- 3*s.d. 99.7%
  
### ROC

  - ROC: Receiver operating characteristic curve
  - AUC: Area under the ROC curve

ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

  - Y - TPR, X - FPR
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

### One Sample T-test

  - z = (p - p_0) / sqrt((p_0 * (1 - p_0)) / n)
  - z = (mu - mu_0) / sqrt(sigma^2 / n)
  
Note:
  - Power of test: 1 - beta
  
### Two Sample T-test

  - https://onlinecourses.science.psu.edu/stat500/node/50
  
### ANOVA

  - Analysis of Variance (ANOVA) is a statistical method used to test differences between two or more means.
  - A hypothesis test that is used to compare the means of two populations is called t-test. A statistical technique that is used to compare the means of more than two populations is known as Analysis of Variance or ANOVA.
  - Test Statistic: Between Sample Variance/Within Sample Variance
  - Multiply t-test will increase the type I (false positive) rate: a null hypothesis is rejected although it is true.
  - If there are only two conditions of the independent variable, doing ANOVA is the same as running a (two-tailed) two-sample t-test.
  
Examples:
  - http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_HypothesisTesting-ANOVA/BS704_HypothesisTesting-Anova_print.html
  
### [Chi-square Test](https://onlinecourses.science.psu.edu/statprogram/node/158/)

  - [This lesson](https://stattrek.com/chi-square-test/independence.aspx) explains how to conduct a chi-square test for independence. The test is applied when you have two categorical variables from a single population. It is used to determine whether there is a significant association between the two variables.

  - For example, in an election survey, voters might be classified by gender (male or female) and voting preference (Democrat, Republican, or Independent). We could use a chi-square test for independence to determine whether gender is related to voting preference.

  - chi-square = sum( (observed_frequency_i - expected_frequency_i)^2 / expected_frequency_i )
  
### K-S Test

https://onlinecourses.science.psu.edu/stat414/node/323

In statistics, the Kolmogorov–Smirnov test (K–S test or KS test) is a nonparametric test of the equality of continuous, one-dimensional probability distributions that can be used to compare a sample with a reference probability distribution (one-sample K–S test), or to compare two samples (two-sample K–S test).
  
### Simpson’s Paradox 

Simpson’s Paradox is a paradox in probability and statistics, in which a trend appears in different groups of data but disappears or reverses when these groups are combined.

### Gaussian Noise and Regularization

  - By adding Gaussian noise to the input, the learning model will behave like an L2-penalty regularizer. 
  - By adding Laplace noise to the input, the learning model will behave like an L1-penalty regularizer.

### Assumptions of linear regression

  - Linear relationship between Y and X1, …, Xp
  - Error Term i.i.d (Independent and identically distributed) 均值为0，近normal 分布(normally distributed residuals)，constant variance和independently distributed. 
	- Error has normal distribution.
	- The errors have zero mean.
	- The errors have the same (but unknown) variance. When this assumption does not hold, the problem is called the heterogeneity or the heteroscedasticity problem. (191/391 Regression Analysis By Example)
	  - Weighted Least Squares sum(w_i * (y_i - beta_0 - beta_1 * x_i)^2)
	  - w_i = 1/sigma_i^2, sigma_i^2 is the variance of epsilon_i.
	  - Ordinary Least Squared sum( (y_i - beta_0 - beta_1 * x_i)^2 )
	- The errors are independent of each other. When this assumption does not hold, we have the autocorrelation problem. (213/391 Regression Analysis By Example)
	  - The first type is only autocorrelation in appearance. It is due to the omission of a variable that should be in the model, such as seasonality.
	  - The second type of autocorrelation may be referred to as pure autocorrelation. The methods of correcting for pure autocorrelation involve a transformation of the data.
  - The predictor variables X1, . . . , Xp, are assumed to be linearly independent of each other; variables are nonrandom; variable values are all measured without error.
  - All observations are equally reliable and have approximately equal role in determining the regression results and in influencing conclusions.

### With or Without Replacement

With Replacement
  - Sampling with replacement is used to find probability with replacement. In other words, you want to find the probability of some event where there’s a number of balls, cards or other objects, and you replace the item each time you choose one. Let’s say you had a population of 7 people, and you wanted to sample 2. Their names are: John, Jack, Qiu, Tina, Hatty, Jacques, Des. You could put their names in a hat. If you sample with replacement, you would choose one person’s name, put that person’s name back in the hat, and then choose another name. The possibilities for your two-name sample are: [John, John], [John, Jack], [John, Qui], … and so on.

Without Replacement
  - Sampling without Replacement is a way to figure out probability without replacement. In other words, you don’t replace the first item you choose before you choose a second.
  
### [Autoregression and Moving Average (AR, MA)](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)

Autoregression
  - x(t) = c + sum( beta_i * x(t - i)) + epsilon(t)
  
Moving Average
  - x(t) = mu + epsilon(t) + sum( q_i * epsilon(t - i))
  
ARIMA
  - https://people.duke.edu/~rnau/411arim.htm
  - ARIMA(p,d,q)
  - Differencing in statistics is a transformation applied to time-series data in order to make it stationary.
  - If d=0:  y(t) = Y(t)
  - If d=1:  y(t) = Y(t) - Y(t-1)
  - If d=2:  y(t) =  (Y(t) - Y(t-1)) - (Y(t-1) - Y(t-2))  =  Y(t) - 2Y(t-1) + Y(t-2)

### Ways to avoid overfitting
  - regularization 
  - dropout
  - cross validation
  - early stopping 
  - ...

### Linear Regression and Correlation Relationship

Linear Regression
  - Y = a + b * X
  - b = Cov(X, Y) / Variance(X) = Cov(X, Y) / Sigma(X)^2
  
Pearson Correlation
  - rho = Cov(X, Y) / Sigma(X) * Sigma(Y)

# Problems

### X,Y ~ N(0,1), correlation of (X+Y, X)?

  - Pearson correlation coefficient formula: rho = Cov(X, Y) / Sigma(X) * Sigma(Y)
  - Cov(X, Y) = E((X - E(X))*(Y - E(Y)))
  - Sigma^2 = E(X^2) - (E(X))^2
  - X ~ N(0, 1), E(X) = 0, Sigma^2 = E(X^2) - (E(X))^2 = 1
  
Answer:
  - sqrt(2) / 2

### Pick up a coin from C1 and C2 with probability of trials p(h1)=.7, p(h2)=.6 and doing 10 trials. And what is the probability that the given coin you picked is C1 given you have 7 heads and 3 tails?

  - p(C1|7H) = p(7H|C1) * p(C1) / p(7H)
  - p(7H) = p(7H|C1)*p(C1) + p(7H|C2)*p(C2)
  - assume p(C1) = p(C2) = 0.5, pick either w/ equal prob.
  - p(7H | C1) = C(10,7)*0.7^7*0.3^3
  - ans = C(10,7)*0.7^7 * 0.3^3 * 0.5 / [ C(10,7)*0.7^7 * 0.3^3 * 0.5 + C(10,7)*0.6^7 * 0.4^3 * 0.5]

### A box contains a red ball, an orange ball, a yellow ball, a green ball, a blue ball and an indigo ball. Billy randomly selects 7 balls from the box (with replacement). What is the expected value for the number of distinct colored balls Billy will select?
  - Let P(i), 0 < i <= 7 be the probability of having i distinct balls after drawing 7 balls.
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
  - To get (1, 2, 3, 4, 5, 6): 1 + 6/5 + 6/4 + 6/3 + 6/2 + 6/1

#### 1000 people, each time select 10 (w/o replacement) 问每个人on average多少次会被抽到?

  - In sampling without replacement, focus on how many iterations are needed. That number will be any count from 1 through 100, each equally likely. 
  - So sampling without replacement, it will be 1/100*(1+2+...+100) = 50.5.

#### 1000 people, each time select 10, (w/ replacement), 问每个人on average多少次会被第一次抽到（1000个人，你每次抽10个人。 每次抽完以后再把这10个人放回去。这样有些人会被抽中1次，有些人会被抽中2次，etc。 比如你抽了10次之后，有一个人Mike, 他第一次被你抽中。现在问，平均下来，一个人被第一次抽中时， 你抽了几次。）

  - p = 1 / 100
  - Expected times to be selected 1/p = 100
  
#### 目问说箱子里面有三种球，红色，蓝色和白色，开始一个一个往出拿，with replacement，这个游戏当你拿到红球的时候就停止了，求问说蓝色球被拿到次数的期望。

  - p = 1/3
  - Expected times: 1/p = 3 
  
### If the probability a car get at least one ticket within an hour is 84%, what is the probability of getting at least one ticket in half an hour?

  - Say the probability of not get a ticket within half an hour is 1-x, then one hour without a ticket is (1-x)*(1-x)=1-0.84. x=0.6
  
or, 

  - It really depends on what model is assumed. However, if the idea is that no matter how long you leave your car there, you have a 16% chance of getting through any given hour unscathed, you can treat it as an exponential decay problem. Let p(t) be the probability that you do not get a ticket in the first t hours. Then p(1)=0.16, p(2)=0.16*0.16 (a 16% chance of making it through the first hour times a 16% chance of making it through the second), and in general p(t)=0.16^t. The probability of not getting a ticket in the first half hour is then p(1/2)=0.16^(1/2)=0.4, and the probability that you do get a ticket in the first half hour is about 1−0.4=0.6.
  
### I throw three (n) darts along a circle, what is the probability all three(n) points are in a semi-circle?

  - Suppose that point i has angle 0 (angle is arbitrary in this problem) -- essentially this is the event that point i is the "first" or "leading" point in the semicircle. Then we want the event that all of the points are in the same semicircle -- i.e., that the remaining points end up all in the upper half plane.

  - That's a coin-flip for each remaining point, so you end up with 1 / 2^(n−1). There's n points, and the event that any point i is the "leading" point is disjoint from the event that any other point j is, so the final probability is n / 2^(n−1) (i.e. we can just add them up).

  - A sanity check for this answer is to notice that if you have either one or two points, then the probability must be 1, which is true in both cases.
  
### 1) A and B have a game. There are 7 games in total, who wins 4 games first will succeed the whole game and then the game ends. Given that A has a probability P to win a single game, and A already lost the first 2 games, what's the probability that A still wins the whole game. 2) Given that someone said A has 80% probability to win the whole game, what is the posterior that A win the whole game.'

  - C(5, 1) * P^4 * (1-P)

### Consider a game with 2 players, A and B. Player A has 8 stones, player B has 6. Game proceeds as follows. First, A rolls a fair 6-sided die, and the number on the die determines how many stones A takes over from B. Next, B rolls the same die, and the exact same thing happens in reverse. This concludes the round. Whoever has more stones at the end of the round wins and the game is over. If players end up with equal # of stones at the end of the round, it is a tie and another round ensues. What is the probability that B wins in 1, 2, ..., n rounds?

  - On the first round, B can win if (A,B) rolls: (1,4), (1,5), (1,6), (2,5), (2,6), (3,6) - so there are 6 out of 36 possibilities where B wins. P(B1) [read, probability that B wins in round 1] = 1/6 On the first round, B can tie if (A,B) rolls: (1,3), (2,4), (3,5), (4,6) - so there are 4 out of 36 possibilities where B ties. If B ties with A, there is a second round, where B wins with probability 15/36, A wins with probability 15/36 = 5/12, and they tie with probability 6/36. So P(B2) = 1/9 * (5/12) After the second round, the game only continues if both players have equal number of points, and the probability of a tie in each game is 1/6, so P(B3) = (1/9) * (1/6) * (5/12) Generally, P(Bn) = (1/9) * (1/6)^(n-2) * (5/12)
  
### 问的是有100个红色的球和100个白色的球，还有两个盒子。问怎么样把所有球装入这两个盒子里，可以得到从这两个箱子里取一个球是红球的概率最大。

  - 一个放1个红球。另一个99个红球，100个白球。这样概率不就是0.5 + 0.5 * 99/199了吗~
  
### Bobo the amoeba has a 25%, 25%, and 50% chance of producing 0, 1, or 2 offspring, respectively. Each of Bobo's descendants also have the same probabilities. What is the probability that Bobo's lineage dies out?

  - The probability of lineage dies out for Bob = probability of lineage dies out for each of Bob’s son’s. 
  - Let’s assume the probability is P, then p = 0.25 + 0.25 * p + 0.5 * (p * p) => p = 0.5
  - That is, Bob’lineage dies out = Bob has not child + Bob has one child * this child’s lineage dies out + Bob has two children * both children’s lineage dies out
  
### There is a building with 100 floors. You are given 2 identical eggs. How do you use 2 eggs to find the threshold floor, where the egg will definitely break from any floor above floor N, including floor N itself.

### 有两个硬币，都是有bias的，但是你不知道具体每个硬币head朝上的概率，现在让你扔100次，扔到head给你1刀，扔到tail付1刀，问怎么设计策略可以赢得最多。

  - UCB : An Optimal Exploration Algorithm for Multi-Armed Bandits 
  
### Expected value of minimum of two uniform random variables

  - 广告bid, 出价最高的公司可以拿到那个广告位置，但是价格是第二高的那个价格。比如说有3个出价，100,50,30。那么出价100的赢了，但只需要出50块。 问题比较简单，如果只有两个人bid，这两人的出价都是Uniform(0,1),那么expected的广告收入说多少？
  - 我说一个价格x_1, 一个x_2, 本质求 min(x_1, x_2)的期望。
  - From [StackExchange](https://math.stackexchange.com/questions/786392/expectation-of-minimum-of-n-i-i-d-uniform-random-variables)
    - By definition probability, F(y) = P(Y ≤ y) = 1 − P(Y > y) = 1 − P(min(X1,…,Xn) > y). 
	- Of course, min(X1,…Xn) > y exactly when Xi > y for all i. 
	- Since these variables are i.i.d., we have F(y) = 1 − P(X1 > y)P(X2 > y)…P(Xn > y) = 1 − P(X1 > y)^n. 
	- From F(y), derivative to get the density function f(y)
	- The expect value E(Y) = integral(y * f(y) * dy)
  - [Another Solution](http://premmi.github.io/expected-value-of-minimum-two-random-variables)
  
### Variance On Different Machines

计算Var(x1,x2,...xn), 但是所有xi分布在k个不同的服务器上。服务器运算力很强，服务器之间不能传数据，只能和本地一台很差的机器相互传输。怎么尽量减少传输数据量。

  - var(X) = E(X^2)-(E(X))^2
  - E(X) & E(X^2)都可以先在服务器上计算一部分，传3个数 (sum(Xi), sum(Xi^2), count(Xi)) 回本地，最后本地再合成。







