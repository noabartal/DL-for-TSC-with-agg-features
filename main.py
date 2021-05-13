from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import time
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import FEATURES
from utils.constants import SELECTION
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS, NORMALIZE, DENSE
from utils.utils import read_all_datasets, create_agg_tsfresh, feature_selection
import tensorflow as tf
import random

# Seed value
seed_value = 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)


def fit_classifier(input_path, features=10, method='cfs', iter=0, dense=32, size=None):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    print(f"number of classes {nb_classes}")
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]

    if size:
        x_train = x_train[:size, :, :]
        y_train = y_train[:size, :]

    if classifier_name.endswith('extension'):

        start_time = time.time()

        x_train_agg, y_train_max, x_val_agg, y_val_max = create_agg_tsfresh(x_train, y_train, x_test, y_test, input_path, size=size)
        if int(x_train_agg.shape[1] * (features / 100.0)) < 3:  # we scale the number of features if it is to low
            features = features * 10

        x_train_agg_filtered, x_val_agg_filtered = feature_selection(x_train_agg, y_train_max,
                                                                     x_val_agg, input_path,
                                                                     features=features, method=method)

        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                                       input_agg=x_train_agg_filtered.shape[1:], verbose=0, dense=dense)
        classifier.fit(x_train, y_train, x_test, y_test, y_true, x_train_agg_filtered, x_val_agg_filtered)
        duration = time.time() - start_time
        print(f'finish all train {duration}')
    else:
        start_time = time.time()

        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=0)
        classifier.fit(x_train, y_train, x_test, y_test, y_true)
        duration = time.time() - start_time

        print(f'finish all train {duration}')


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, input_agg=10, dense=32):

    if classifier_name == 'fcn_extension':
        from classifiers import fcn_extension
        return fcn_extension.Classifier_CNN_TSFRESH(output_directory, input_shape, nb_classes, verbose, input_agg=input_agg, dense=dense)
    if classifier_name == 'resnet_extension':
        from classifiers import resnet_extension
        return resnet_extension.Classifier_RESNET_TSFRESH(output_directory, input_shape, nb_classes, verbose, input_agg=input_agg, dense=dense)
    if classifier_name == 'inception_extension':
        from classifiers import inception_extension
        return inception_extension.Classifier_INCEPTION_TSFRESH(output_directory, input_shape, nb_classes, verbose, input_agg=input_agg, dense=dense)

    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = ''

if sys.argv[1] == 'run_all':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    root_dir = sys.argv[2]
    start_iter = int(sys.argv[5])
    num_of_iters = int(sys.argv[6])
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(start_iter, start_iter + num_of_iters):
                print('\t\titer', iter, flush=True)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)
                for dense in DENSE:
                    trr_d = trr
                    if dense != 32:
                        trr_d += '_dense_' + str(dense)
                        if 'dense' not in classifier_name:
                            continue
                    for feature in FEATURES:
                        print('\t\tfeature', feature, flush=True)
                        trr_fe = trr_d
                        if feature != 0:
                            if 'extension' not in classifier_name:
                                continue
                            params = '_f_' + str(feature)
                            if NORMALIZE:
                                params += 'norm'
                            trr_fe += params
                        elif 'extension' in classifier_name:
                            continue

                        for i, met in enumerate(SELECTION):
                            print('\t\tselection', met, flush=True)

                            trr_fe_se = trr_fe

                            if 'extension' not in classifier_name:
                                if i > 0:
                                    continue
                            else:
                                trr_fe_se += '_' + met

                            tmp_output_directory = root_dir + 'results/' + classifier_name + '/' + archive_name + trr_fe_se + '/'
                            for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                                print('\t\t\tdataset_name: ', dataset_name, flush=True)

                                output_directory = tmp_output_directory + dataset_name + '/'

                                create_directory(output_directory)

                                fit_classifier(root_dir + 'archives/' + archive_name + '/' + dataset_name + '/',
                                               features=feature, method=met, iter=iter, dense=dense)

                                print('\t\t\t\tDONE', flush=True)

                                # the creation of this directory means
                                create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper('', filename='results/inception_my_inception_compares.csv')
elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())

# experiment with one data set to check the runtime
elif sys.argv[1] == 'Wafer':
    archive_name = sys.argv[3]
    datasets_dict = read_all_datasets(root_dir, archive_name)
    classifier_name = sys.argv[4]
    dataset_name = 'Wafer'
    for size in range(100, 1001, 100):
        print(f'size {size}')
        tmp_output_directory = root_dir + 'results/' + classifier_name + '/' + archive_name + 'size' + str(size) + '/'

        # if dataset_name in UNIVARIATE_DATASET_NAMES_2018_small:
        #     continue

        print('\t\t\tdataset_name: ', dataset_name, flush=True)

        output_directory = tmp_output_directory + dataset_name + '/'

        create_directory(output_directory)

        fit_classifier(root_dir + 'archives/' + archive_name + '/' + dataset_name + '/',
                       features=3, method='ufs', iter=0, dense=None, size=size)

else:
    # this is the code used to launch an experiment on a dataset
    root_dir = sys.argv[2]
    archive_name = sys.argv[3]
    classifier_name = sys.argv[4]
    start_iter = int(sys.argv[5])
    num_of_iters = int(sys.argv[6])

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), flush=True)
    # if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    #     import weka.core.jvm as jvm
    #     jvm.start(system_cp=True, packages=True, system_info=True)
    print('\tarchive_name', archive_name, flush=True)
    print('\tclassifier_name', classifier_name, flush=True)

    datasets_dict = read_all_datasets(root_dir, archive_name)

    for iter in range(start_iter, start_iter + num_of_iters):
        print('\t\titer', iter, flush=True)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)
        for dense in DENSE:
            trr_d = trr
            if dense != 32:
                trr_d += '_dense_' + str(dense)
                if 'dense' not in classifier_name:
                    continue
            for feature in FEATURES:
                print('\t\tfeature', feature, flush=True)
                trr_fe = trr_d
                if feature != 0:
                    if 'extension' not in classifier_name:
                        continue
                    params = '_f_' + str(feature)
                    if NORMALIZE:
                        params += 'norm'
                    trr_fe += params
                elif 'extension' in classifier_name:
                    continue

                for i, met in enumerate(SELECTION):
                    print('\t\tselection', met, flush=True)

                    trr_fe_se = trr_fe

                    if 'extension' not in classifier_name:
                        if i > 0:
                            continue
                    else:
                        trr_fe_se += '_' + met

                    tmp_output_directory = root_dir + 'results/' + classifier_name + '/' + archive_name + trr_fe_se + '/'
                    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                        # if dataset_name in UNIVARIATE_DATASET_NAMES_2018_small:
                        #     continue

                        print('\t\t\tdataset_name: ', dataset_name, flush=True)

                        output_directory = tmp_output_directory + dataset_name + '/'

                        create_directory(output_directory)

                        fit_classifier(root_dir + 'archives/' + archive_name + '/' + dataset_name + '/',
                                       features=feature, method=met, iter=iter, dense=dense)

                        print('\t\t\t\tDONE', flush=True)

                        # the creation of this directory means
                        create_directory(output_directory + '/DONE')

    if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
        jvm.stop()
