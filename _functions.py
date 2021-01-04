
# %% Preliminaries

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline


# Paths for plotting
MAIN_PATH = r"C:\Users\DEPMORA1\Documents\Projects\aktuarvereinigung"
OUTPUT_PATH = rf"{MAIN_PATH}\output"

# %% Function for creating day/ month/ year and delta for date series


# Incident Date/ Policy Bind Date
def date_variable_creation(column, data):
    datetime_series = pd.to_datetime(data.loc[:, column], format="%Y-%m-%d")
    year_series = datetime_series.dt.year
    month_series = datetime_series.dt.month
    day_series = datetime_series.dt.day
    delta_series = (datetime_series.max() - datetime_series).dt.days
    return year_series, month_series, day_series, delta_series


# %% Data Types Check


def categorical_check(column, data):
    """
    This function checks whether the series is a categorical variable. We work here with the assumptions that if
    there are less than 5 percent unique values that we have a categorical variable. Furthermore we test whether
    we face a binary or a string variable. The latter would always be a categorical variable.
    """
    series = data.loc[:, column].dropna()
    bool_series_cat = series.nunique() / len(series) < 0.05
    bool_binary = list(set(series) - {0, 1}) == []
    bool_object = data.dtypes[column] == object
    return (bool_series_cat and not bool_binary) or bool_object


def binary_check(column, data):
    """This function checks whether the series is a binary"""
    series = data.loc[:, column].dropna()
    bool_binary = list(set(series) - {0, 1}) == []
    return bool_binary


def float_check(column, data):
    """This function checks whether the variable is a float variable. Same logic applied as in the cat function"""
    series = data.loc[:, column].dropna()
    bool_float = series.nunique() / len(series) > 0.05
    bool_object = data.dtypes[column] == object
    return bool_float and not bool_object


# %% Plotting Functions


def nominal_ordinal_plots(column, data):
    """This function plots pie charts for all nominal and ordinal columns"""
    nom_ord_value_counts = data.loc[:, column].value_counts()

    fig, axs = plt.subplots(figsize=(10, 10))
    nom_ord_value_counts.plot(kind="pie", ax=axs)
    path = rf"{OUTPUT_PATH}\nominal_ordinal\{column}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def categorical_plots(column, data):
    """This function shows the percentage of observations of all categories which belong to target Y or N"""
    column_w_target = ["fraud_reported", column]
    counts = data.loc[:, column_w_target].groupby(column_w_target).size().unstack("fraud_reported")
    column_counts = counts.sum(axis="columns")
    props = counts.div(column_counts, axis="index")

    fig, axs = plt.subplots(figsize=(10, 10))
    props.plot.barh(stacked=True, ax=axs)
    path = rf"{OUTPUT_PATH}\categorical\{column}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def float_plots(column, data):
    """This function plots histograms of each float column separated for both cases of the target variable"""
    df_float = data.loc[:, ["fraud_reported", column]]
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.histplot(x=column, data=df_float, hue="fraud_reported")
    path = rf"{OUTPUT_PATH}\floats\{column}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def float_outliers(column, data):
    """This function plots a boxplot of each float variable and shows the z-score to give an outlier indication"""
    series = data.loc[:, column]
    max_z_score = max((series - np.mean(series)) / np.std(series))

    fig, axs = plt.subplots(figsize=(10, 10))
    sns.boxplot(series, ax=axs)
    plt.suptitle(f"The maximal z-score is {round(max_z_score, 2)}")
    path = rf"{OUTPUT_PATH}\outliers\{column}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()


# %% a


def less_than_six_categories(column, data):
    """Since columns with more than 2 but less than 6 categories are dummied, we have to find those"""
    series = data.loc[:, column].astype(str) # Since NaN should be counted as a category we have to make it string
    num_cats = len(series.value_counts())
    if (num_cats <= 5) and (num_cats > 1):
        return True
    else:
        return False


# %% a


def pca_scatter_plot(data, target, name):
    """This function applies PCA on all features in order to then plot a two dimensional scatter plot"""
    pca = PCA(n_components=2)
    scaled_data = MinMaxScaler().fit_transform(data)
    df_pca = pd.DataFrame(pca.fit_transform(scaled_data), columns=["PCA_0", "PCA_1"])
    df_pca.loc[:, "target"] = target

    fig, axs = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x="PCA_0", y="PCA_1", data=df_pca, hue="target")
    path = rf"{OUTPUT_PATH}\SMOTE\{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()


# %% Project Evaluation Methods


def monetary_score_calc(model_name, y_true, y_pred, test_claim_amount):
    """This function calculates how much money was saved from the prediction and how must was lost."""
    dict_monetary_score = {
        "sum_saved_money": sum(test_claim_amount[(y_true == 1) & (y_pred == 1)]),
        "sum_lost_money": sum(test_claim_amount[(y_true == 1) & (y_pred == 0)]),
        "num_unnecessary_checks": sum((y_true == 0) & (y_pred == 1)),
        "sum_if_nothing_checked": sum(test_claim_amount[(y_true==1)])
    }
    df_monetary_scores = pd.DataFrame.from_dict(dict_monetary_score, orient="index", columns=["scores"])
    df_monetary_scores.loc[:, "model_name"] = model_name
    return df_monetary_scores


def score_calculation(model_name, y_true, y_pred):
    """This function evaluates the performance of the predictor through different metrices."""
    dict_scores = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    df_scores = pd.DataFrame.from_dict(dict_scores, orient="index", columns=["scores"])
    df_scores.loc[:, "model_name"] = model_name
    return df_scores


def calc_feature_importance(model, columns):
    """Helper function of prediction_evaluation. This function calculates feature importances and aligns
    them with the corresponding column names"""
    feature_importances = model.best_estimator_.steps[1][1].feature_importances_
    df_feature_importance = pd.DataFrame(feature_importances.reshape(1, -1),
                                         columns=columns)
    return df_feature_importance


def calc_best_parameters(model):
    """Helper function of prediction_evaluation. This function calculates which parameter of the grid_search
    were the best ones and returns those"""
    dict_best_params = model.best_params_
    df_model_best_params = pd.DataFrame.from_dict(dict_best_params, orient="index", columns=["scores"])
    return df_model_best_params


def prediction_evaluation(original_data, x, y, model, model_name, random_state=1,
                          grid_search=False, columns_list=None):
    """
    This function splits, trains and evaluates the data. The latter step is done through all the helper function
    above this function. Furthermore it allows for having a gridsearch and feature importance. For the latter option
    we have to provide the column names in order to match the relative importance with the variable

    :param original_data: DataFrame with all the column names - Used to extract the claim amount
    :param x: Array of processed data
    :param y: Array/Series of the target in encoded form
    :param model: Model or pipeline
    :param model_name: str of the model in order to map the results
    :param random_state: int
    :param grid_search: bool whether we would like to obtain the best parameters
    :param columns_list: list of all the column names
    :return: Varying list of dataframes of all results
    """

    # Obtain the related total claim amount
    series_total_claim_amount = original_data.loc[:, "total_claim_amount"]
    total_claim_train, total_claim_test = train_test_split(series_total_claim_amount, test_size=0.25,
                                                           random_state=random_state)

    # Obtain Predictions
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Evaluations of monetary terms and standard evaluation methods
    eval_df = score_calculation(model_name, y_test, y_pred)
    monetary_df = monetary_score_calc(model_name, y_test, y_pred, total_claim_test)
    list_returns = [eval_df, monetary_df]

    # Add the best parameters if we throw a grid-search at the model
    if grid_search:
        list_returns += [calc_best_parameters(model)]

    # If desired we can also add the feature importance
    if columns_list is not None:
        list_returns += [calc_feature_importance(model, columns_list)]

    return list_returns


def appending_temp(temps, mains):
    """This function helps appending all results from every random state"""
    assert len(temps) == len(mains), "Temp and Main files are not equally long"
    for i in range(len(mains)):
        mains[i] = mains[i].append(temps[i])
    return mains


def plot_boxplot(dataframe, png_name):
    """This function plots all results from every random state in a boxplot"""
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.boxplot(x="method", y="scores", hue="model_name", data=dataframe, ax=axs)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45)
    path = rf"{OUTPUT_PATH}\{png_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close()


# %% Column Names extraction


def onehotencoder_namechange(original_list, list_binary_variables):
    """
    This function is a add-on to the feature name extraction below. It helps to also transform the
    binary variables
    """
    for i, bin_var in enumerate(list_binary_variables):
        original_list = [x.replace(f"__x{i}_", f"__{bin_var}_") for x in original_list]
    return original_list


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


def train_best_model(pipeline, reshaped_best_params, processed_data, y):
    """
    This function takes the hyperparameter which were deemed the best most of the time and
    trains the model on all data
    """
    df_mode = reshaped_best_params.mode()
    pipeline.set_params(
        gradientboostingclassifier__learning_rate=df_mode.loc[0, "gradientboostingclassifier__learning_rate"],
        gradientboostingclassifier__max_depth=int(df_mode.loc[0, "gradientboostingclassifier__max_depth"]),
        gradientboostingclassifier__min_samples_leaf=int(df_mode.loc[0, "gradientboostingclassifier__min_samples_leaf"]),
        gradientboostingclassifier__n_estimators=int(df_mode.loc[0, "gradientboostingclassifier__n_estimators"]),
        smote__sampling_strategy=df_mode.loc[0, "smote__sampling_strategy"]
    )

    pipeline.fit(processed_data, y)
    return pipeline
