# Deep Learning for Time Series Classification with aggregated features
This repository contains the models and experiments for the paper titled "Deep learning for time series classification with aggregated features"  which is currently under review.
This paper is part of my MSc thesis.
![architecture](https://github.com/noabartal/DL-for-TSC-with-agg-features/blob/master/images/diagram.png)

## Data 
We used two publicly available repositories and one unique data set.
* The [UCR archive](http://www.timeseriesclassification.com/dataset.php), which contains the 128 univariate time series datasets (we use 116 of them). 
* The [MTS archive](http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html), which contains the 12 multivariate time series datasets.
* The [SHRP2 dataset](https://insight.shrp2nds.us/), which contains multivariate vehicle sensors data that used for driver identification task.

## Code 
The code is a clone of [ISMAIL FAWAZ git repository](https://github.com/hfawaz/dl-4-tsc), with some adjustments.
* The [main.py](https://github.com/noabartal/DL-for-TSC-with-agg-features/tree/master/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/noabartal/DL-for-TSC-with-agg-features/tree/master/utils) folder contains the necessary functions to read the datasets and visualize the plots and also the parameters for the experiments.
* The [CFSmethod](https://github.com/noabartal/DL-for-TSC-with-agg-features/tree/master/CFSmethod) contains the Correlation based Feature Selection method based on [ZixiaoShen git repository](https://github.com/ZixiaoShen/Correlation-based-Feature-Selection/tree/master/CFSmethod)
* The [ec_feature_selection](https://github.com/noabartal/DL-for-TSC-with-agg-features/tree/master/ec_feature_selection) contains the Feature Ranking and Selection via Eigenvector Centrality method for feature selection based on [OhadVolk git repository](https://github.com/OhadVolk/ECFS) with adjustments for multiclass task.
* The [classifiers](https://github.com/noabartal/DL-for-TSC-with-agg-features/tree/master/classifiers) folder contains six python files, three for the deep neural network architectures without our extension 
and three for the deep neural network architectures with our extension.

To run a model on the archive list of datasets (configured in utils/constants) you should issue the following command: 
```
python3 main.py kind_of_run directory_path archive_name classifier_name beginning_iter_number number_of_iterations
```

Specific example:
```
python3 main.py run_archive DL-for-TSC-with-agg-features\ UCRArchive_2018 inception_extension 0 10
```

which means we are launching the [inception_extension](https://github.com/noabartal/DL-for-TSC-with-agg-features/blob/master/classifiers/inception_extension.py) model on the univariate UCR archive (see [constants.py](https://github.com/noabartal/DL-for-TSC-with-agg-features/blob/master/utils/constants.py) for 10 iterations).



##### Packages and the used vesions can be found in the requirements.txt file


## Results
The results on the [UCR_archive](http://www.timeseriesclassification.com/dataset.php) 116 datasets for
 ResNet, InceptionTime, FCN, HIVE-COTE, ResNet with aggregated features, InceptionTime with aggregated features, 
 FCN with aggregated features are presented in the [UCR_116_all_classifiers](https://github.com/noabartal/DL-for-TSC-with-agg-features/blob/master/UCR_116_all_classifiers.csv) table.

The results on the [MTS archive](http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html) 12 datasets for 
ResNet, FCN, ResNet with aggregated features, FCN with aggregated features are presented in the [MTS_12_all_classifiers](https://github.com/noabartal/DL-for-TSC-with-agg-features/blob/master/MTS_12_all_classifiers.csv) table.

The results on the [SHRP2 dataset](https://insight.shrp2nds.us/) for 
 ResNet, InceptionTime, FCN, ResNet with aggregated features, InceptionTime with aggregated features, 
 FCN with aggregated features are presented in the [SHRP2_all_classifiers](https://github.com/noabartal/DL-for-TSC-with-agg-features/blob/master/SHRP2_all_classifiers.csv) table.

## Reference
TBA



