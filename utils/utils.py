from builtins import print
import numpy as np
import pandas as pd
import matplotlib
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from scipy import stats
from utils.constants import NORMALIZE
from xgboost import XGBClassifier
import time

import seaborn as sns
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import utils.constants
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator
from CFSmethod.CFS import fcbf
from ec_feature_selection import ECFS

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS, FEATURES, SELECTION
from utils.constants import ITERATIONS
from utils.constants import MTS_DATASET_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import interp1d
from scipy.io import loadmat
from tsfresh import select_features, extract_features, extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from matplotlib import pyplot


def df_from_3d_np(data):
    data_out = np.column_stack((np.repeat(np.arange(data.shape[0]), data.shape[1]),
                                np.tile(np.arange(data.shape[1]), data.shape[0]),
                                data.reshape(data.shape[0] * data.shape[1], -1)))

    data_df = pd.DataFrame(data_out, columns=['index', 'time'] + list(np.arange(data_out.shape[1] - 2)))

    return data_df


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + 'results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def create_agg_tsfresh(x_train, y_train, x_val, y_val, input_path, size=None):

    y_train = pd.DataFrame(y_train).idxmax(axis=1)
    y_val = pd.DataFrame(y_val).idxmax(axis=1)
    if os.path.exists(input_path + 'agg_train.csv') and os.path.exists(input_path + 'agg_val.csv') and size is None:

        x_train_filtered = pd.read_csv(input_path + 'agg_train.csv', index_col=0)
        x_val_filtered = pd.read_csv(input_path + 'agg_val.csv', index_col=0)

        x_train_filtered = x_train_filtered.loc[:, x_train_filtered.var() != 0]
        x_val_filtered = x_val_filtered[x_train_filtered.columns]

    else:
        x_train_df = df_from_3d_np(x_train)
        x_val_df = df_from_3d_np(x_val)

        x_train_df = x_train_df.fillna(0)
        x_val_df = x_val_df.fillna(0)
        # start_time = time.time()
        x_train_extracted = extract_features(
            x_train_df,
            column_id='index', column_sort='time', default_fc_parameters=EfficientFCParameters())
        # duration = time.time() - start_time
        # print(f'feature extraction {duration}')
        if 'mts_archive' in input_path:
            x_train_sel = select_features(x_train_extracted, y_train, n_jobs=0)

            # if not enough features, take larger set
            if x_train_sel.shape[1] < 300:
                X_best = SelectKBest(f_classif, k='all').fit(x_train_extracted, y_train)
                ufs_scores = pd.DataFrame(X_best.scores_, index=x_train_extracted.columns, columns=['score']).sort_values(
                    by=['score'], ascending=False)
                x_train_sel = x_train_extracted[ufs_scores.iloc[:300].index]

            x_train_extracted = x_train_sel

        x_train_extracted = x_train_extracted.dropna(axis='columns')

        x_train_extracted.to_csv(input_path + f'agg_train.csv')
        y_train.to_csv(input_path + f'y_train.csv')

        x_val_filtered = pd.read_csv(input_path + 'agg_val.csv', index_col=0)

        x_train_filtered = x_train_extracted.loc[:, x_train_extracted.var() != 0]
        x_val_filtered = x_val_filtered[x_train_filtered.columns]

        y_val.to_csv(input_path + 'y_test.csv')

    return x_train_filtered, y_train, x_val_filtered, y_val


def feature_selection(x_train, y_train, x_test, input_path, features, method):
    """
    select features from the aggregated features set
    :param x_train:
    :param y_train:
    :param x_test:
    :param input_path:
    :param features:
    :param method: weka or cfs (two different implementations of correlation feature selection)  / ufs (ANOVA based) / ecfs
    :return:
    """
    if method == 'cfs':
        x_train_filtered, x_val_filtered = feature_selection_cfs(x_train, y_train, x_test, input_path, features)
    elif method == 'weka':
        x_train_filtered, x_val_filtered = feature_selection_weka(x_train, y_train, x_test, input_path, features)
    elif method == 'ufs':
        x_train_filtered, x_val_filtered = feature_selection_ufs(x_train, y_train, x_test, input_path, features)
    # elif method == 'xgboost':
    #     x_train_filtered, x_val_filtered = feature_selection_xgboost(x_train, y_train, x_test, input_path, features)
    elif method == 'ecfs':
        x_train_filtered, x_val_filtered = feature_selection_ecfs(x_train, y_train, x_test, input_path, features)
    else:
        print('method not exist')
        return x_train, x_test
    if NORMALIZE:
        scaler = MinMaxScaler(feature_range=(0, 1))

        x_train_filtered.loc[:, x_train_filtered.columns] = scaler.fit_transform(x_train_filtered.values)
        x_val_filtered.loc[:, x_val_filtered.columns] = scaler.transform(x_val_filtered.values)

    print(f'after feature selection {x_train_filtered.shape}')

    return x_train_filtered, x_val_filtered


def feature_selection_ufs(x_train, y_train, x_test, input_path, features):
    percent = int(x_train.shape[1] * (features / 100.0))
    if not os.path.exists(input_path + f'ufs_scoresa.csv'):
        x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]
        X_best = SelectKBest(f_classif, k=percent).fit(x_train, y_train)

        ufs_scores = pd.DataFrame(X_best.scores_, index=x_train.columns, columns=['score']).sort_values(by=['score'], ascending=False)
        ufs_scores.to_csv(input_path + "ufs_scores.csv")
    else:
        ufs_scores = pd.read_csv(input_path + f'ufs_scores.csv', index_col=0)

    x_train_filtered = x_train[ufs_scores.iloc[:percent].index]
    x_val_filtered = x_test[ufs_scores.iloc[:percent].index]

    return x_train_filtered, x_val_filtered


def feature_selection_ecfs(x_train, y_train, x_test, input_path, features):
    percent = int(x_train.shape[1] * (features / 100.0))

    if not os.path.exists(input_path + f'ecfs_summary.csv'):
        ecfs = ECFS(n_features=percent)

        # ufs = pd.read_csv(input_path + "ufs_scores.csv", index_col=0)
        # x_train = x_train[ufs.iloc[:1200].index]
        ecfs.fit(X=x_train, y=y_train, alpha=0.5, positive_class=1, negative_class=0)
        summary = pd.DataFrame({'Feature': x_train.columns, 'Ranking': ecfs.ranking, 'MI': ecfs.mutual_information,
                                'Fisher Score': ecfs.fisher_score})
        summary = summary.sort_values(by='Ranking')
        summary.to_csv(input_path + f'ecfs_summary.csv')

    else:
        summary = pd.read_csv(input_path + f'ecfs_summary.csv', index_col=0)
    x_train_filtered = x_train.loc[:, summary.iloc[:percent]['Feature']]
    x_val_filtered = x_test.loc[:, summary.iloc[:percent]['Feature']]
    return x_train_filtered, x_val_filtered


def feature_selection_xgboost(x_train, y_train, x_test, input_path, features):
    percent = int(x_train.shape[1] * (features / 100.0))

    if not os.path.exists(input_path + f'feature_importance_XGBOOST.csv'):

        model = XGBClassifier()
        model.fit(x_train, y_train)
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, x_train.columns)),
                                   columns=['Value', 'Feature']).sort_values(by="Value", ascending=False).reset_index(
            drop=True)

        feature_imp.to_csv(input_path + 'feature_importance_XGBOOST.csv')

    else:
        feature_imp = pd.read_csv(input_path + 'feature_importance_XGBOOST.csv')

    x_train_filtered = x_train.loc[:, feature_imp.iloc[:percent]['Feature']]
    x_val_filtered = x_test.loc[:, feature_imp.iloc[:percent]['Feature']]

    return x_train_filtered, x_val_filtered


def feature_selection_weka(x_train, y_train, x_test, input_path, features):
    percent = int(x_train.shape[1] * (features / 100.0))
    if not os.path.exists('Weka'):
        os.mkdir('Weka')

    if not os.path.exists(input_path + f'selected_features_weka_{features}.csv'):
        x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]
        sava_data = x_train.copy()
        sava_data.columns = [str(a) + "a" for a in range(sava_data.shape[1])]
        sava_data['target'] = y_train
        sava_data.to_csv('Weka/train_weka_format.csv', index=False)

        from weka.attribute_selection import ASEvaluation, AttributeSelection,ASSearch
        from weka.core.converters import Loader, Saver
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file('Weka/train_weka_format.csv', class_index='last')

        search = ASSearch(classname="weka.attributeSelection.GreedyStepwise", options=["-C", "-R", "-N", f"{percent}"])
        evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval",
                                 options=["-P", "1", "-E", "1", "-L"])
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(data)
        ranked_attributes = pd.DataFrame(attsel.ranked_attributes, columns=['Feature', 'Rank'])
        ranked_attributes['Feature'] = ranked_attributes['Feature'].astype(int)
        set_of_features = ranked_attributes.loc[:percent - 1, 'Feature']

        x_train.iloc[:, set_of_features].to_csv(input_path + f'selected_features_weka_{features}.csv')
        selected_features = x_train.iloc[:, set_of_features].columns
    else:
        selected_features = pd.read_csv(input_path + f'selected_features_weka_{features}.csv', index_col=0).columns

    x_train_filtered = x_train.loc[:, selected_features]
    x_val_filtered = x_test.loc[:, selected_features]

    return x_train_filtered, x_val_filtered


def feature_selection_cfs(x_train, y_train, x_test, input_path, features):
    th = 0
    percent = int(x_train.shape[1] * (features / 100.0))

    if not os.path.exists(input_path + f'correlation_dict{th}.csv'):

        correlation_dict, remove_history = fcbf(x_train, y_train, threshold=th, base=2, is_debug=False)

        pd.DataFrame(correlation_dict).to_csv(input_path + f'correlation_dict{th}.csv')

    correlation_dict = pd.read_csv(input_path + f'correlation_dict{th}.csv', index_col=0)
    correlation_dict = correlation_dict.iloc[:percent, :]
    x_train_filtered = x_train.loc[:, correlation_dict['0']]
    x_val_filtered = x_test[list(x_train_filtered.columns)]

    return x_train_filtered, x_val_filtered


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'mts_archive':
        file_name = cur_root_dir + 'archives/' + archive_name + '/' + dataset_name + '/'
        x_train = np.load(file_name + 'x_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    elif archive_name == 'UCRArchive_2018':
        root_dir_dataset = cur_root_dir + 'archives/' + archive_name + '/' + dataset_name + '/'
        df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    else:
        file_name = cur_root_dir + 'archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = readucr(file_name + '_TRAIN')
        x_test, y_test = readucr(file_name + '_TEST')
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict


def read_all_datasets(root_dir, archive_name, split_val=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []

    if archive_name == 'mts_archive':

        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = cur_root_dir + 'archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'UCRArchive_2018':
        for dataset_name in DATASET_NAMES_2018:
            root_dir_dataset = cur_root_dir + 'archives/' + archive_name + '/' + dataset_name + '/'

            df_train = pd.read_csv(root_dir_dataset + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(root_dir_dataset + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = np.nanstd(x_train, axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - np.nanmean(x_train, axis=1, keepdims=True)) / std_

            std_ = np.nanstd(x_test, axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - np.nanmean(x_test, axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    else:
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = cur_root_dir + 'archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict


def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x


def transform_mts_to_ucr_format():
    mts_root_dir = 'archives/mtsdata/'
    mts_out_dir = 'archives/mts_archive/'
    for dataset_name in MTS_DATASET_NAMES:
        # print('dataset_name',dataset_name)

        out_dir = mts_out_dir + dataset_name + '/'

        # if create_directory(out_dir) is None:
        #     print('Already_done')
        #     continue

        a = loadmat(mts_root_dir + dataset_name + '/' + dataset_name + '.mat')
        a = a['mts']
        a = a[0, 0]

        dt = a.dtype.names
        dt = list(dt)

        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'testlabels':
                y_test = a[i].reshape(max(a[i].shape))

        # x_train = a[1][0]
        # y_train = a[0][:,0]
        # x_test = a[3][0]
        # y_test = a[2][:,0]

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        print(dataset_name, 'max', max_length, 'min', min_length)
        print()
        # continue

        x_train = transform_to_same_length(x_train, n_var, max_length)
        x_test = transform_to_same_length(x_test, n_var, max_length)
        os.mkdir(out_dir)
        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def perform_wilcoxon(new_method, old_method):
    wilcoxon_result = stats.wilcoxon(new_method, old_method)
    print(wilcoxon_result)


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def generate_results_csv(output_file_name, root_dir):
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    for classifier_name in CLASSIFIERS:
        print(classifier_name)
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir, archive_name)
            for it in range(0, ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                for feature in FEATURES:
                    curr_archive_name_fe = curr_archive_name
                    if feature != 0:
                        if 'extension' not in classifier_name:
                            continue
                        params = '_f_' + str(feature)
                        # if NORMALIZE:
                        #     params += 'norm'
                        curr_archive_name_fe += params
                    elif 'extension' in classifier_name:
                        continue
                    without_norm = curr_archive_name_fe
                    for norm in ['norm', '']:
                        curr_archive_name_fe = without_norm + norm
                        for i, met in enumerate(SELECTION):
                            curr_archive_name_fe_se = curr_archive_name_fe

                            if 'extension' not in classifier_name:
                                if i > 0:
                                    continue
                            else:
                                curr_archive_name_fe_se += '_' + met
                                # curr_archive_name = curr_archive_name
                            for dataset_name in datasets_dict.keys():
                                output_dir = root_dir + 'results/' + classifier_name + '/' \
                                             + curr_archive_name_fe_se + '/' + dataset_name + '/' + 'df_metrics.csv'
                                # print(output_dir)
                                if not os.path.exists(output_dir):
                                    continue
                                df_metrics = pd.read_csv(output_dir)
                                df_metrics['classifier_name'] = classifier_name
                                df_metrics['archive_name'] = archive_name
                                df_metrics['dataset_name'] = dataset_name
                                df_metrics['feature_selection_method'] = met
                                df_metrics['features'] = feature
                                df_metrics['normalize'] = norm

                                res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    # aggreagte the accuracy for iterations on same dataset
    # res = pd.DataFrame({
    #     'accuracy': res.groupby(
    #         ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    # }).reset_index()

    res_agg = res.groupby(['classifier_name', 'archive_name', 'dataset_name', 'feature_selection_method', 'features', 'normalize'])[
        'accuracy'].agg(['mean', 'median', 'std', 'max'])
    res_agg['res'] = (res_agg['mean'] * 100).round(1).astype(str) + '(' + \
                     (res_agg['std'] * 100).round(1).astype(str) + ')'
    res_agg.reset_index().to_csv(root_dir + f'results/results_agg_{"".join(CLASSIFIERS)}.csv', index=False)
    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs_t_leNet(output_directory, hist, y_pred, y_true, duration):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]
    print(f"index_best_model {index_best_model}, best test performance {df_metrics['accuracy']}")

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def visualize_filter(root_dir):
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')

    return 1


def viz_perf_themes(root_dir, df):
    df_themes = df.copy()
    themes_index = []
    # add the themes
    for dataset_name in df.index:
        if dataset_name in utils.constants.dataset_types.keys():
            themes_index.append(utils.constants.dataset_types[dataset_name])
        else:
            themes_index.append("unknown")

    themes_index = np.array(themes_index)
    themes, themes_counts = np.unique(themes_index, return_counts=True)
    df_themes.index = themes_index
    df_themes = df_themes.rank(axis=1, method='min', ascending=False)
    df_themes = df_themes.where(df_themes.values == 1)
    df_themes = df_themes.groupby(level=0).sum()
    df_themes['#'] = themes_counts

    for classifier in df_themes.columns:
        df_themes[classifier] = df_themes[classifier] / df_themes['#'] * 100
    df_themes = df_themes.round(decimals=1)
    df_themes.to_csv(root_dir + 'tab-perf-theme.csv')


def viz_perf_train_size(root_dir, df):
    df_size = df.copy()
    train_sizes = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCRArchive_2018')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(datasets_dict[dataset_name][0])
        train_sizes.append(train_size)

    train_sizes = np.array(train_sizes)
    bins = np.array([0, 100, 400, 800, 99999])
    train_size_index = np.digitize(train_sizes, bins)
    train_size_index = bins[train_size_index]

    df_size.index = train_size_index
    df_size = df_size.rank(axis=1, method='min', ascending=False)
    df_size = df_size.groupby(level=0, axis=0).mean()
    df_size = df_size.round(decimals=2)

    print(df_size.to_string())
    df_size.to_csv(root_dir + 'tab-perf-train-size.csv')


def viz_perf_classes(root_dir, df):
    df_classes = df.copy()
    class_numbers = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCRArchive_2018')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(np.unique(datasets_dict[dataset_name][1]))
        class_numbers.append(train_size)

    class_numbers = np.array(class_numbers)
    bins = np.array([0, 3, 4, 6, 8, 13, 9999])
    class_numbers_index = np.digitize(class_numbers, bins)
    class_numbers_index = bins[class_numbers_index]

    df_classes.index = class_numbers_index
    df_classes = df_classes.rank(axis=1, method='min', ascending=False)
    df_classes = df_classes.groupby(level=0, axis=0).mean()
    df_classes = df_classes.round(decimals=2)

    print(df_classes.to_string())
    df_classes.to_csv(root_dir + 'tab-perf-classes.csv')


def viz_perf_length(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCRArchive_2018')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths = np.array(lengths)
    bins = np.array([0, 81, 251, 451, 700, 1001, 9999])
    lengths_index = np.digitize(lengths, bins)
    lengths_index = bins[lengths_index]

    df_lengths.index = lengths_index
    df_lengths = df_lengths.rank(axis=1, method='min', ascending=False)
    df_lengths = df_lengths.groupby(level=0, axis=0).mean()
    df_lengths = df_lengths.round(decimals=2)

    print(df_lengths.to_string())
    df_lengths.to_csv(root_dir + 'tab-perf-lengths.csv')


def viz_plot(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCRArchive_2018')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths_index = np.array(lengths)

    df_lengths.index = lengths_index

    plt.scatter(x=df_lengths['fcn'], y=df_lengths['resnet'])
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0, xmax=1.05)
    # df_lengths['fcn']
    plt.savefig(root_dir + 'plot.pdf')


def viz_for_survey_paper(root_dir, filename='results-uea.csv'):
    df = pd.read_csv(root_dir + filename, index_col=0).iloc[:, :2]
    # df = df.T
    # df = df.groupby(['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean().reset_index()
    # df = df.round(decimals=4)
    # df.set_index()
    # get table performance per themes
    viz_perf_themes(root_dir,df)

    # get table performance per train size
    viz_perf_train_size(root_dir,df)

    # get table performance per classes
    viz_perf_classes(root_dir,df)

    # get table performance per length
    viz_perf_length(root_dir,df)

    # get plot
    viz_plot(root_dir, df)


def viz_cam(root_dir):
    import tensorflow.keras as keras
    import sklearn
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'

    if dataset_name == 'Gun_Point':
        save_name = 'GunPoint'
    else:
        save_name = dataset_name
    max_length = 2000
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_test = datasets_dict[dataset_name][3]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes:
        plt.figure()
        count = 0
        c_x_train = x_train[np.where(y_train == c)]
        for ts in c_x_train:
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0, :, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                # if (y < -2.2).any():
                #     continue
                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.0)
                if dataset_name == 'Gun_Point':
                    if c == 1:
                        plt.yticks([-1.0, 0.0, 1.0, 2.0])
                    else:
                        plt.yticks([-2, -1.0, 0.0, 1.0, 2.0])
                count += 1

        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels([100,75,50,25,0])
        plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + save_name + '-class-' + str(int(c)) + '.png',
                    bbox_inches='tight', dpi=1080)
