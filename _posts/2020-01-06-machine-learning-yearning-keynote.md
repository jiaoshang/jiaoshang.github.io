---
title: Machine Learning Yearning Keynote
tags: machine_learning
layout: article
article_header:
  type: overlay
  theme: dark
  background_image:
      src: /assets/images/2020-01-06-machine-learning-yearning-keynote/wood-3240764_1920.jpg
---

<!--more-->

## Train, Dev(validation) & Test set

> **Training set**: Which you run your learning algorithm on.
>
> **Dev(validation) set**: Which you use to tune parameters, select features, and make other decisions regarding the learning algorithm. 
>
> **Test set​:** Which you use to evaluate the performance of the algorithm, but not to make any decisions regarding what learning algorithm or parameters to use.

### Dev set & Test set

- Don't assume the training distribution is the same the your test distribution. Try to choose dev and test sets that reflect what you **ultimately want to perform well on**, rather than whatever data you happen to have for training.
- Dev and test set should have the **same distribution**.
- The dev set should be **large enough to detect differences between algorithms that you are trying out**. For mature and important applications, for example, advertising, web search, and product recommendations, dev set size could be large enough in order to detect even smaller improvements.
- Test set size should be large enough to **give high confidence in the overall performance of your system**.
- There is no need to have excessively large dev/test sets beyond what is needed to evaluate the performance of your algorithms.

### Evaluation metric

- Having a ​**single-number evaluation metric​** such as accuracy allows you to sort all your models according to their performance on this metric, and quickly decide what is working best.
-  If there are multiple goals be cared about, consider combining them into a single formula or defining satisficing and optimising metrics. (Set **some thresholds** for the other requirements like running time, model size. Then try to **optimise the metric given those constraints**.)

### Dev/test set and Metric

<div class="w3-container">
  <img src="/assets/images/2020-01-06-machine-learning-yearning-keynote/ml-model-building-loop.png" class="image image--xl" alt="Machine Learning Model Building Loop" align="middle">
  <p>Machine Learning Model Building Loop</p>
</div>

- Having a dev set and metric allows you to very quickly detect which ideas are successfully giving you small (or large) improvements, and therefore lets you quickly decide what ideas to keep refining, and which ones to discard. It can speed up the machine learning system building iterations.
- It is quite common to change dev/test sets or evaluation metrics during a project. Having an initial dev/test set and metric quickly helps you **iterate quickly**.
- If you ever find that the dev/test sets or metric are no longer pointing your team in the right direction, it’s not a big deal! Just change them and make sure your team knows about the new direction. The following points might be a sign that the dev/test set or metric need to be changed.
  - **The actual distribution you need to do well on is different from the dev/test sets, get new dev/test sets**.
  - **You have overfit to the dev set, get more dev set data**. The process of repeatedly evaluating ideas on the dev set causes your algorithm to gradually “overfit” to the dev set. If dev set performance is much better than test set performance, it is a sign that you have overfit to the dev set.
  - **The metric is measuring something other than what the project needs to optimise, change the metric**.

## Basic Error Analysis
When you start a new project, especially if it is in an area in which you are not an expert, it is hard to correctly guess the most promising directions. So don’t start off trying to design and build the perfect system. Instead build and train a basic system as quickly as possible. Then use error analysis to help you identify the most promising directions and iteratively improve your algorithm from there.
> **Error Analysis**: The process of examining dev set examples that your algorithm misclassified, so that you can understand the underlying causes of the errors. This can help you prioritise projects

Error analysis does not produce a rigid mathematical formula that tells you what the highest priority task should be. You also have to take into account how much progress you expect to make on different categories and the amount of work needed to tackle each one.

### Best Practices for Carrying Out Error Analysis
- Evaluating multiple ideas in parallel. Create a spreadsheet with error categories. And come up with a few ideas.
- Cleaning up mislabeled dev and test set examples by human.
  - It is not uncommon to start off tolerating some mislabeled dev/test set examples, only later to change your mind as your system improves so that the fraction of mislabeled examples grows relative to the total set of errors.
  - Whatever process you apply to fixing dev set labels, remember to do the same for the test set labels to keep dev and test sets has the same distribution.
  - **NOTE:** If you decide to improve the label quality, consider double-checking both the labels of examples that your system misclassified as well as labels of examples it correctly classified. It is possible that both the original label and your learning algorithm were wrong on an example. If you fix only the labels of examples that your system had misclassified, you might introduce bias into your evaluation. But because it is easier in practice to check only the misclassified examples, bias does creep into some dev sets. In most case, the bias is acceptable if you are interested only in developing a product or application.
- Split a large dev set into **Eyeball dev set** and **Blackbox dev set** to know when the dev set is overfitted by manual error analysis.
  - Eyeball dev set: Look into it to do error analysis. It should be large enough to give you a sense of your algorithm's major error categories. But it should also be determined by how many examples  you have time to analyse manually.
  - Blackbox dev set: Evaluate trained model, select algorithm, tune hyper-parameters. It should have enough data to **tune hyper-parameters and select among models**.
  - Use a small dev set as an Eyeball dev set entirely, but keep in mind that the risk of overfitting is greater.
  - Since you will gain intuition about Eyeball dev set example, Eyeball dev set will be overfitted faster
    - Eyeball dev set is overfitted: If the performance on the Eyeball dev set improving much more rapidly than performance on Blackbox dev set. In this case, 
      - Find a new Eyeball dev set by by moving more examples from the Blackbox dev set in the Eyeball dev set, or
      - Acquiring new labeled data 
  - Examining an Eyeball dev set **won't be helpful for a task that even humans can't do well**, since it's harder to figure out why the algorithm didn't predict correctly.

## Bias & Variance

Two major sources of error in machine learning,

> **Bias**: The algorithm's error rate on the training set
> 
> **Variance**: The difference between the dev error and the training error.
> 
> **Optimal error rate (Unavoidable bias)**:  The error rate of an optimal algorithm (e.g. the best speech system in the world has error rate 14%, so the optimal error rate is 14%). In statistics, the optimal error rate is also called Bayes error rate, or Bayes rate
>
> **Avoidable bias**: The difference between the training error and the optimal error rate.

### Example

| Case | Training Error | Dev Error | Bias & Variance                   | Performance                |
| ---- | -------------- | --------- | --------------------------------- | -------------------------- |
| 1    | 1%             | 11%       | Low bias & **High variance**      | Overfitting                |
| 2    | 15%            | 16%       | **High bias** & Low variance      | Underfitting               |
| 3    | 15%            | 30%       | **High bias** & **High variance** | Overfitting & Underfitting |
| 4    | 0.5%           | 1%        | Low bias & Low variance           | Great                      |

### Techniques for reducing avoidable bias

- Increase the model size: e.g. neurons, layers
- Modify input features based on insights from error analysis
  - It's useful to do an error analysis on the Eyeball training set for high bias issue
  - It could help with both bias and variance
- Reduce or eliminate regularisation​: e.g. L1/L2 regularisation, dropout
- It will increase variance
- Modify model architecture​ 
- Add more training data **is not helpful**

### Techniques for reducing variance
- Add more training data
- Add regularisation​
  - It will increase bias
- Add early stopping: 
 - It will increase bias
- Feature selection to decrease number/type of input features
- Decrease the model size
 - **Use with caution**, it might increase bias.
- Modify input features based on insights from error analysis
 - It could help with both bias and variance 
- Modify model architecture

## Learning Curves

> Learning curve plot your **dev set and training set error** against the **number of training examples**.

It make us more confident to diagnose the model issue by examining both the training error curve and dev error curve on the same plot.

### Interpreting learning curves

#### High Bias

<div class="w3-container">
  <img src="/assets/images/2020-01-06-machine-learning-yearning-keynote/high_bias_learning_curve.png" class="image image--xl" alt="High bias learning curves (From Machine Learning Yearning Andrew Ng)" align="middle">
  <p>High bias learning curves (From Machine Learning Yearning Andrew Ng)</p>
</div>

Adding more training data makes training performance worse, and dev error is usually higher than training error. So, it's almost impossible to make the model reach desired performance by adding more training data.

#### High Variance

<div class="w3-container">
  <img src="/assets/images/2020-01-06-machine-learning-yearning-keynote/high_variance_learning_curve.png" class="image image--xl" alt="High variance learning curves (From Machine Learning Yearning Andrew Ng)" align="middle">
  <p>High variance learning curves (From Machine Learning Yearning Andrew Ng)</p>
</div>

Training error is relatively low, so the bias is small. Dev error is much larger than the training error, thus, the variance is large. By adding more training data probably help the model reach desired performance.

#### High Bias & High Variance

<div class="w3-container">
  <img src="/assets/images/2020-01-06-machine-learning-yearning-keynote/high_bias_variance_learning_curve.png" class="image image--xl" alt="High bias & high variance learning curves (From Machine Learning Yearning Andrew Ng)" align="middle">
  <p>High bias & high variance learning curves (From Machine Learning Yearning Andrew Ng)</p>
</div>

The training error is large and dev error is even larger, so it's a high bias and high variance issue. It's better to find a way reduce both bias and variance.

### Plotting Learning Curves
When your plotted learning curves are too noisy to find the underlying trends. Try,
- For an **extremely small** training set sample: Randomly select several groups of training sample with same size, compute and plot the average training and dev error.
- For an **imbalance or multi-class** training set: Try to select a 'balanced' subset instead of random selecting. Try to make the fraction of examples from each class is as close as possible to the overall fraction in the original training set.

Plotting learning curves might be **computationally expensive**. When the computational cost of all additional models is significant, you might train models with 1000, 2000, 4000, 6000 and 10000 examples (instead of evenly spacing out the training set size on a linear scale), it should give you a clear sense of the trend.

## Comparing to Human-Level Performance
Many ML systems aims to automate things that humans do well. There are several reasons building an ML system is easier on a task that people are good at,
- Ease of obtaining data from human labelers
- Error analysis can draw on human intuition
- Use human-level performance to estimate the optimal error rate and also set a desired error rate

There are some tasks that human can't do well,  and computers already surpass the performance of most people.  With these cases, it's easy to run into the following problems,
- It is harder to obtain labels
- Human intuition is harder to count on
- It is hard to know what the optimal error rate and reasonable desired error rate is

### Define Human-Level Performance
- Use the a group pf experts' performance as the human-level performance, but it's always expensive to label all the data with an expert, perhaps it's good enough to ask junior to label the majority of cases, only bring harder cases to expert
- If the current model performance is 40% error,  it doesn't matter that your data is labeled by and get intuition from a junior (10% error), an expert (5% error), or a group of experts (2% error)
- If the current model's performance is 10% error, it makes more sense to use data that is labeled by a group pf experts 

### Surpassing Human-Level Performance
- Even if the model's **average performance on dev/test set** is already surpassing human-level performance. As long as there are dev set examples where **humans are right and model is wrong**, you can still make benefit by obtaining labels from human, drawing on human intuition and using human-level performance as desired performance target
- Progress is usually slower on problems where model performance already surpass human-level performance, while progress is faster when models are still trying to catch up to humans

## Training & Testing on Different Distributions
In the big data era, it's easy to get a huge training set from Internet. In some cases, they can provide a lot of information even its distribution is different with the target (dev/test set). e.g. Use cat images from the web as training set and use the cat images photoed by customer as dev/test set.

 - It gives your model more examples of your task. Your model can apply some of the knowledge acquired from Internet images to photos. 
 - It forces the model (neural network) to expand some of its capacity to learn about properties that are specific to Internet images (such as higher resolution). If the properties differ greatly from photos, it will “use up” some of the representational capacity of the neural network. Thus there is less capacity for recognising data drawn from the distribution of photos, it could hurt the model's performance.

For the 2nd effect, 
- If you have computational capacity to build a big enough neural network, then the 2nd effect is not a serious concern, since the model has enough capacity to learn from both Internet images and photos. 
- If you can't build a big enough neural network, then making the training data matching dev/test set distribution is important. Thus, if you think some data has no benefit, they should be left out for computational reasons. 

Only use `inconsistent` data when there is some x —> y mapping that works well for both types of data.

### Weighting Data
Suppose the ratio of Internet examples (200,000) and photo examples (5,000) is 40:1, to build a neural network with cost huge computational resources.
Take as square error optimisation objective, thus the model tries to optimise: