# OptimSeed

The source code and seed word sets used for [Seed Word Selection for Weakly-Supervised Text Classification with Unsupervised Error Estimation](paper/paper.pdf), to appear in NAACL-HLT SRW 2021.

<p align="center"><img width="100%" src="img/fig1.jpg"/></p>

## Step I: Keyword Expansion
Please see [keyword_expansion.ipynb](keyword_expansion.ipynb). It takes an initial seed word for each category and an unlabeled corpus to expand seed words for each category. Since there're many experiments, I used yaml config files (under the [configs](configs) folder) to manage the experiments. You can ignore the other fields and update the following parameters in the config file. 

```
categories = config['categories']  # the list of categories 
seed_words = config['seed_words']  # the initial seed words (usually the category name)
output_file = config['kw_file']  # the output keyword file after keyword expansion
corpus_path = config['train_corpus_path']  # the input corpus
```

In the notebook I added scripts to load each dataset from their original form (individual files under folder/csv). If you experiment on your own dataset, you might modify from an existing example or write your own.

## Step II: Train Weakly-Supervised Classifiers with Candidate Seed Words

We use the Generalized Expectation (GE) Java implementation provided in the [MALLET library](http://mallet.cs.umass.edu/ge-classification.php). The code is relatively straight-forward to set up and run. We do not provide the Java code to train interim classifiers. You can either follow the instructions in the MALLET library to use GE or replace it with another weakly-supervised text classification model.

## Step III: Unsupervised Error Estimation

**Note:** I implemented the Bayesian error estimation (BEE) algorithm in Numpy following the paper [Estimating Accuracy from Unlabeled Data: A Bayesian Approach](http://proceedings.mlr.press/v48/platanios16.html). I tried to follow the paper as closely as possible. However, due to my limited knowledge in Bayesian statistics, I cannot guarantee that the code is error-free. Later I realized that the original author of the BEE paper published his source code [here](https://github.com/eaplatanios/makina). You can consider using his repo instead if you're familiar with Java.

The main class for BEE is [bee.py](bee.py). 

```
def __init__(self, labeling_matrix, num_iters=50, init_method='sampling', filter_estimators=False, filter_by_std=False):
        """ Initialize the BEE model

        :param labeling_matrix: the prediction matrix [num_samples, num_estimators]. Each entry is either 0 or 1
        :param num_iters: the number of Gibbs sampling iterations
        :param init_method: [maj, sampling]
        """
```

The only parameter you need to pass to it is the labeling matrix, which contains the binary predictions from interim classifiers. All other parameters you can leave as default. The model performs inference upon initialization and you can access the following attributes after it's done.

```
bee.true_labels: the inferred latent label for each example.
bee.error_rates: the estimated error rate for each interim classifier (the inverse of the accuracy). 
```

## Seed Words in the Paper

We provide the full list of seed words for all datasets under the [seedwords](seedwords) folder. The seed word files are named according to the following convention: 

```
[dataset name]-[category A]-[category B]-[seed word set]
```

An example `AGNews-business-sports-all` means it's from AGNews dataset the Business-Sports classification task and using all keywords after keyword expansion. Note that the naming of the seed word sets are slightly different. Please refer to the mapping below:

| Seed word set in this repo | Seed word set in the paper (Table 3 & 4) |
|----------------------------|------------------------------------------|
| seed                       | cate                                     |
| all                        | -                                        |
| ours                       | ours                                     |
| manual                     | seed                                     |


## Datasets

All datasets used in this work are public datasets published in previous work.

| Dataset | Link |
|----------------------------|------------------------------------------|
| AGNews                       | [Link](https://github.com/yumeng5/WeSTClass/tree/master/agnews)                                     |
| NYT                        | [Link](https://github.com/yumeng5/WeSHClass/tree/master/nyt)                                        |
| Yelp                       | [Link](https://github.com/yumeng5/WeSTClass/tree/master/yelp)                                     |
| IMDB                     | [Link](https://ai.stanford.edu/~amaas/data/sentiment/)                                     |

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{jin2021seed,
  title={Seed Word Selection for Weakly-Supervised Text Classification with Unsupervised Error Estimation},
  author={Jin, Yiping and Bhatia, Akshay and Wanvarie, Dittaya},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Student Research Workshop},
  year={2021}
}
```
