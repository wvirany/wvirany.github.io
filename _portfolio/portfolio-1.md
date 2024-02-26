---
title: "Machine Learning to Predict Microbiome-Phenotype Associations in the Human Gut"
excerpt: "Using machine learning to predict diseases in humans based on the content of their gut microbiome.<br><br/><img src='/images/Microbiome-Project-Photo.png' width=500>"
collection: portfolio
date: 2023-12-15
---

#### Walter Virany

---


![microbiome_cover_photo](/images/Microbiome-Project-Photo.png)

The following is a brief report of the project. The full report and source code can be viewed on [GitHub](https://github.com/wvirany/microbiome)

The human microbiome is a data rich ecosystem which is generally known to have strong associations with host phenotypes (i.e., the set of observable characteristics of an organism). However, the extents of these relationships are not yet deeply understood. Until recently, the computing capabilities necessary to thoroughly analyze human microbiome data were severely lacking, but due to novel advancements in sequencing technologies and data analysis techniques, there now exist a plethora of data, along with robust statistical tools, which can be leveraged to gain valuable insights into the impact of the human microbiome on overall health. Methods which can accurately detect disease early in developmental stages have the potential to revolutionize personalized and preventative medicine. In this project, I explore state-of-the-art machine learning methods for processing human microbiome data, which I then use to predict the likelihood of someone devloping IBD. I also perform feature selection to identify which biomarkers are most likely to contribute to the condition. I find that IBD can be attributed to approximately 15-20 different species of bacteria found in the human gut.


## Data Loading and Preprocessing

The data I will be will be analyzing was taken as a part of the Integrative Human Microbiome Project$^1$ (HMP2), which includes 1627 stool samples from medical patients. Each patient was classified with IBD or as a control (i.e., healthy). This phenotype is the target variable. The features are the relative species abundances found in each stool sample. This is the proportion of each species recorded in the sample with respect to the others. First, I access the data from `curatedMetagenomicData()`$^2$, a package within R's Bioconductor library.

I begin by performing exploratory data analysis. First, I plot a histogram of the relative species abundance in three healthy samples and three samples corresponding to patients with IBD. I also normalize the data by dividing each feature by its maximum, scaling each feature to a value between zero and one.

<p align="center">
  <img src="/images/histogram_healthy.png">
</p>
<p align="center">
  <img src="/images/histogram_ibd.png">
</p>

<!-- ![healthy_histogram](/images/histogram_healthy.png)
![ibd_histogram](/images/histogram_ibd.png) -->

Immediately we can see that the sampels with IBD seem to have a lower species diversity than the healthy samples.


## Feature Selection: Applying Lasso and ENet to Optimize Support Vector Machine

The resulting dataset is a sparse matrix of relative species abundance, with a large amount of features. My next goal is to see how each model performs on different subsets of the features. In the long run, this will help me make biological insights as to which species are most commonly linked to IBD. First, I implement a base SVM estimator to test whether or not there is a relationship between the features and the target variable. I do this by implementing `GridSearchCV()`, which performs a 5-fold stratified cross-validation over a given parameter space for the SVM. A radial kernel is used, and the regularization parameter $C$ and width parameter $\gamma$ were chosen from the sets $\{2^{-5}, 2^{-4}, \dots, 2^{15}\}$, and $\{2^{-15}, 2^{-14}, \dots, 2^3\}$, respectively. This was a costly process on the full data set, so the code is commented out here. The optimal parameters were determined to be:

$$
\begin{align*}
C = 2048, \\
\gamma = .5
\end{align*}
$$

The ROC curve corresponding to the base SVM estimator is shown:

<p align="center">
  <img src="/images/roc_curve1.png">
</p>

This figure shows the ROC curve for the base SVM estimator. Now, I want to perform feature selection on the dataset and see how the model performs on various subsets of the original features. So, I implement two different feature selection methods; Lasso and Elastic Net Regularization. Then, I assess each model's accuracy on different subsets of the features.

First, I implement `LassoCV()`, which searches for the optimal alpha parameter in $\{10^{-4}, 10^{-3.5}, \dots, 10^{.5}\}$, as described in Pasolli et al.$^3$

Similarly, I implement `ENetCV()`, which searches for the optimal `alpha` parameter as before, as well as the optimal `l1 ratio` parameter in $\{.1, .5, .7, .9, .95, .99, 1\}$.

Interestingly, ENetCV returned an L1 ration of .1, which leans more towards an L$^2$ norm for feature selection.

I want to determine which features to include in my model, so I will train/test the model while varying the percentage of features included in the process based on their levels of importance:

<p align="center">
  <img src="/images/prediction_vs_percentage.png">
</p>

It seems like the model with features chosen by Lasso consistently outperforms the one with features chosen by ENet. Furthermore, even with just 10% of the most important features included, the Lasso model still performs well. So, from this point forward I will use the Lasso model.

<p align="center">
  <img src="/images/lasso_coeffs.png">
</p>


This figure demonstrates how the Lasso coefficients change as a function of the regularization parameter. The legend indicates the names of the features included in the model, which are bacterial species found in the gut microbiome. The optimal regularization parameter that is chosen by `LassoCV()` is shown as a vertical line at $\alpha = 10^{-3.5}$.

Next, I implement an SVM on these Lasso coefficients. The following figure shows the resulting ROC curve:

<p align="center">
  <img src="/images/roc_curve2.png">
</p>


## Random Forests

The next step is to construct a random forest and compare its accuracy to the SVM. The process for constructing the RF is outlined in [3], which I reiterate here:

* The parameters are chosen as
  * Number of trees: 500
  * Criterion: Gini impurity
  * Number of features considered at each split: $m = \sqrt{p}$, where $p$ is the total number of predictors
  * `class_weight = balanced`, to account for the imabalance between # of case samples vs. controls (i.e., there are more patients with IBD in the dataset than healthy patients)
* `GridSearchCV()` is performed in an attempt to achieve more optimal paramters; however, no significant improvement was made, so the original parameters were kept
* An implicit feature selection is performed using the impurity-based feature importance. The steps for this process are:
  1. The RF is trained on the whole dataset
  2. The features were ranked according to the impurity-based importance
  3. The RF is retrained on the top $k$ features, where $k$ is chosen from $\{5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200\}$
  4. The number of features that maximizes the accuracy is chosen
  5. The final model is retrained on this subset of features


The following shows the ROC curve for the Random Forest, along with the SVM estimators for comparison.


<p align="center">
  <img src="/images/roc_curve3.png">
</p>

The confusion matrices for each estimator are also shown:

<p align="center">
  <img src="/images/confusion_matrices.png">
</p>

Evidently, the RF tends to perform the best. This could be due to interactions between the features, which the RF accounts for by decorrelating the trees. Moreover, the RF accounts for the imbalance between the classes. The IBD class has about 3 times as many samples as the healthy class. There also seems to be a certain amount of variability on each run. So, I run each model 20 times and average the AUC, F1, and precision scores.


## Discussion

Overall, the random forest model tends to perform the best across all metrics. This could be due to a number of reasons; the foremost being the class imbalance between healthy patients and those with IBD, as well as the correlation between features. The RF is robust to both of these scenarios. Furthermore, it is found that IBD can be predicted from a subset of only 15-20 different species. It is also interesting to note that IBD is generally linked to a deficiency in biodiversity in the human digestive tract. Looking at the histograms of the unprocessed data (Figures 1 and 2), we see that there generally tends to be less diversity in the IBD sampels as opposed to the healthy samples. This is consistent with our biological knowledge. It is also interesting to note that certain features identified here have also been linked in other studies to healthy vs. unhealthy eating habits. $^4$


## Next Steps

For future steps in the project, I would like to explore a number of ideas:



* Perform cross-validation of results across different studies
* Generalize results to larger datasets
* Perform analysis on different diseases
* Explore different models for classification
  * One publication describes a comprehensive analysis of deep learning methods for similar tasks. $^5$
  * Another study attempts to use convolutional neural networks by embedding information about relative species abundance in 2D images. $^6$
  
Evidently, there are a lot of interesting avenues to explore from here.


## References

  1. Lloyd-Price, J., Arze, C., Ananthakrishnan, A.N. et al. Multi-omics of the gut microbial ecosystem in inflammatory bowel diseases. Nature 569, 655–662 (2019). doi.org/10.1038/s41586-019-1237-9.

  2. Pasolli E, Schiffer L, Manghi P, Renson A, Obenchain V, Truong D,
  Beghini F, Malik F, Ramos M, Dowd J, Huttenhower C, Morgan M, Segata N, Waldron L (2017). “Accessible, curated metagenomic data through ExperimentHub.” Nat. Methods, 14(11), 1023–1024. ISSN 1548-7091, 1548-7105, doi:10.1038/nmeth.4468.

  3. Pasolli E, Truong DT, Malik F, Waldron L,
  Segata N (2016) Machine Learning Meta-analysis of
  Large Metagenomic Datasets: Tools and Biological
  Insights. PLoS Comput Biol 12(7): e1004977.
  doi:10.1371/journal.pcbi.1004977

  4. Medawar, E., Haange, SB., Rolle-Kampczyk, U. et al. Gut microbiota link dietary fiber intake and short-chain fatty acid metabolism with eating behavior. Transl Psychiatry 11, 500 (2021). doi.org/10.1038/s41398-021-01620-3

  5. LaPierre, Nathan, et al. ‘MetaPheno: A Critical Evaluation of Deep Learning and Machine Learning in Metagenome-Based Disease Prediction’. Methods, vol. 166, 2019, pp. 74–82, https://doi.org10.1016/j.ymeth.2019.03.003.

  6. Nguyen, Thanh Hai, et al. ‘Disease Classification in Metagenomics with 2D Embeddings and Deep Learning’. arXiv [Cs.CV], 2018, http://arxiv.org/abs/1806.09046. arXiv.