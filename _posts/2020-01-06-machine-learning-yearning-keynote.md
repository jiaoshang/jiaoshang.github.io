---
title: Machine Learning Yearning Keynote
tags: machine_learning
layout: article
mathjax: true
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

$$
\min\limits_{\theta} \sum\limits_{(x,y)\in Internet} (h _{\theta} (x)- y)^2 + \min\limits_{\theta} \sum\limits_{(x,y)\in Photo} (h _{\theta} (x)- y)^2
$$


But by weighting Internet examples less, set 𝛽​=1/40, the model would give equal weight to Internet data and photo data, 

$$
\beta\min\limits_{\theta} \sum\limits_{(x,y)\in Internet} (h _{\theta} (x)- y)^2 +  \min\limits_{\theta} \sum\limits_{(x,y)\in Photo} (h _{\theta} (x)- y)^2
$$

then you don’t have to build as massive a neural network to make sure the algorithm does well on both types of tasks. 

This type of re-weighting is needed **only** when you suspect the additional data (Internet Images) has a very **different distribution** than the dev/test set, or if the additional data is **much larger** than the data that came from the same distribution as the dev/test set (photos).

### Generalising from The Training Set to The Dev Set

If the model is trained on a training set that its distribution is quite different with dev/test set's distribution, and the performance on dev/test set is worse than you expected, it might be wrong in,

1. It does not do well on the training set. This is the problem of high **(avoidable) bias** on the training set distribution.
2. It does well on the training set, but does not generalise well to previously unseen data drawn from the same distribution as the training set.​ This is **high variance**.
3. It generalises well to new data drawn from the same distribution as the training set, but not to data drawn from the dev/test set distribution. This is **​data mismatch​**, since it is because the training set data is a poor match for the dev/test set data.

In order to figure out which issue the model suffers from, it's good to split training set into 2 subsets, then there are four sets,

- Training set: This is the data that the algorithm will learn from. It does not have to be drawn from the same distribution as what we really care about (the dev/test set distribution).
- Training dev set: This data is drawn from the same distribution as the training set This is usually smaller than the training set; it only needs to be large enough to evaluate and track the progress of our learning algorithm.
- Dev set: This is drawn from the same distribution as the test set, and it reflects the distribution of data that we ultimately care about doing well on. 
- Test set: This is drawn from the same distribution as the dev set.

Now you can evaluate,
- Training error, by evaluating on the training set.
- The algorithm’s ability to generalise to new data drawn from the training set distribution, by evaluating on the training dev set.
- The algorithm’s performance on the task you care about, by evaluating on the dev and/or test sets.

#### Examples

|                                                | Distribution A: Internet + Photos | Distribution: Photos |
| ---------------------------------------------- | --------------------------------- | -------------------- |
| Human Level                                    | Human Level Error ~= 0%           |                      |
| Error on examples algorithm has trained on     | Training Error ~= 10%             |                      |
| Error on examples algorithm has not trained on | Training-Dev Error ~=11%          | Dev-Test Error ~=20% |

- Avoidable Bias = Training Error - Human Level Error ~= 10%
- Variance = Training-Dev Error - Training Error ~= 1%
- Data Mismatch = Dev-Test Error - Training-Dev Error ~= 9%

### Addressing Data Mismatch
- Try to understand what properties of the data differ between the training and the dev set distributions
- Try to find more training data that better matches the dev set examples that your algorithm has trouble with
- Unfortunately, there are no guarantees in this process. If you don't have any way to get more training data that better match the dev set data, you might not have a clear path towards improving performance.

### Artificial Data Synthesis
In some circumstances, it's hard to collect a lot of data that you care about (dev/test set) to train your model. It might be easier to do artificially synthesising to generate a huge dataset that reasonably matches the dev/test set. e.g. road noise audio clips + people speaking in quiet room audio = people speaking in noisy road audio

- It is sometimes easier to create synthetic data that appears realistic to a person than it is to create data that appears realistic to a computer. (Generate 1000 hours audio with 1 hour noise audio, model would overfit the 1 hour noise, but when a person listen to the audio,  it's hard for him/her to recognise this issue)
- When synthesising data, put some thought into whether you’re really synthesising a representative set of examples. Try to avoid giving the synthesised data properties that makes it possible for a learning algorithm to distinguish synthesised from non-synthesised examples.
- Data synthesis might takes weeks that are close enough to the actual distribution. But if it succeeds, you can suddenly have a much larger training set than before.


## Debugging Inference Algorithms
### Optimisation Verification Test

Given some input $$x$$,  you know how to compute $$Score_x(y)$$ that indicates how good a response ​y​ is to an input ​x.​ Furthermore, you are using an approximate algorithm to try to find $$ \underset{y}{\operatorname{arg max}} Score_x(y) $$ , but the whole system doesn't perform good.


You can use optimisation verification test method to figure out what went wrong,

Suppose $$y^*$$ is the “correct” output but the algorithm instead outputs $$y_ {out}$$.​ Then the key test is to measure whether $$Score​_x(​y^*) > Score​_x(​ y​_{out})$$​ . There are 2 possibilities, 

1. **Search algorithm problem**​. The approximate search algorithm (beam search) failed to find the value of $​y$​ that maximises $$Score_x(y)$$.
  
2. **Objective (scoring function) problem**.​ Our estimates for $$Score_x(y) = P(​y\|​x)$$ were inaccurate. In particular, our choice of $$Score_x(y)$$ failed to recognise which one is the correct transcription.


- If $$ Score​_x(​y^*) > Score​_x(​y​_{out})​$$, In this case, your learning algorithm has correctly given $$y^*$$ a higher score than $$y_{out}$$.​ Nevertheless, our approximate search algorithm chose S​out r​ather than $$y^*$$. This tells you that your approximate search algorithm is failing to choose the value of $$x$$ that maximises $$Score_x(​y​)$$. In this case, the Optimisation Verification test tells you that you have a search algorithm problem and should focus on that. For example, you could try increasing the beam width of beam search.
- If $$Score​_x(​y^*) ≤ Score_x(​y_{​out})$$​, In this case, you know that the way you’re computing $$Score​_x(​·)$$ is at fault. t is failing to give a strictly higher score to the correct output ​$$y^*$$​ than the incorrect ​yout.​ The Optimisation Verification test tells you that you have an objective (scoring) function problem. Thus, you should focus on improving how you learn or approximate $$Score_x(y)$$ for different sentences $$​y$$.

Note: $$y^*$$ doesn't need to be the optimal output, so long as $$y^*$$  is a superior output to the performance of the current system output. 

To apply the Optimisation Verification test in practice, you should examine the errors in your dev set. For each error, you would test whether $$Score​_x(​y^*) > Score​_x(​y_{​out})$$​ . Each dev example for which this inequality holds will get marked as an error caused by the optimisation algorithm. Each example for which this does not hold ($$Score​_x(​y^*) ≤ Score​_x(​y_{​out})​$$) gets counted as a mistake due to the way you’re computing $$Score​_x(​·)$$

## End-to-end Deep Learning
> End-to-end learning algorithm is using a single learning algorithm to replace a pipeline system (it can contains rules, hand-engineered components, non-ML algorithms). The input of the algorithm can be a 'raw' data, and the output is the target for the pipeline system.

### Pros and Cons of End-to-end Learning
#### Pros
- The non-ML components limit the potential performance of the system
- End-to-end learning systems tend to do well when there is a lot of labeled data for “both ends”—the input end and the output end
- End-to-end deep learning can directly learn ​target​ that are much more complex than a number, e.g. text, image, audio

#### Cons
- Having more hand-engineered components generally allows a system to learn with less data
- The hand-engineered knowledge “supplements” the knowledge the algorithm acquires from data. When we don’t have much data, this knowledge is useful

### Choosing Pipeline Components
- Data Availability: In some cases, there might be not enough data available for training an end-to-end algorithm. But if there is a lot of data available for training "intermediate modules" of a pipeline, it's reasonable to consider using a pipeline with multiple stages. (All available data can be used to train the intermediate modules)
- Task Simplicity: Try to build a pipeline where each component is a relatively "simple" function that can therefore be learned from only a modest amount of data. (If a complex task can be broken down into several simpler sub-tasks, by resolving the sub-tasks explicitly, the algorithm can get some prior knowledge, that can help it learn a task more efficiently, as input)

## Error Analysis by Parts
> By carrying out ​error analysis by parts​, you can try to attribute each mistake the algorithm makes to one (or sometimes several) of the n parts of the pipeline. Look at the output of each part, and see if you can figure out which one made a mistake.

### Attributing error to one part
> Run an experiment in each component with 'perfect' input, with this analysis on the misclassified dev set data,  you can attribute each error to one component. It allows you to estimate the fraction of errors due to each component of the pipeline, and decide the main focus.

// TODO Give example of p111

The components of an ML pipeline should be ordered according to a Directed Acyclic Graph (DAG), meaning that you should be able to compute them in some fixed left-to-right order, and later components should depend only on earlier components’ outputs. So long as the mapping of the components order follows the DAG ordering, then the error analysis will be fine.

Carrying out error analysis on a learning algorithm is like using data science to analyse an ML system’s mistakes in order to derive insights about what to do next. At its most basic, error analysis by parts tells us what component(s) performance is (are) worth the greatest effort to improve.

### Error Analysis by Parts & Comparison to Human-Level Performance
If the project aims to automate something that human can do, the human-level performance can be used as benchmark to do the error analysis process. If one of the components is far from human-level performance, you can have a good case to focus on improving the performance of that component. 

### Spotting a flawed ML Pipeline
> If each individual component of the ML pipeline performs well, but the overall pipeline's performance is bad. This usually means that the pipeline is flawed and needs to be redesigned. Error analysis can also help you understand if you need to redesign your pipeline.

It could be the inputs of some component do not contain enough information, you should rethink what other information is needed.