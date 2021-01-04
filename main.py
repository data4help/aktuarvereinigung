
# Problem also done by the DeutscheAktuarvereinigung: https://github.com/DeutscheAktuarvereinigung/Data_Science_Challenge_2020_Betrugserkennung
# Data obtained from here: https://www.kaggle.com/buntyshah/auto-insurance-claims-data

# %% Preliminaries

#** Packages
# Standard
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Imbalance
import imblearn
from imblearn.over_sampling import SMOTE
# Predictions
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
# Own functions
import _functions as self_func

# Reloading
import importlib
importlib.reload(self_func)

# Paths - Can be commented out for different user
MAIN_PATH = r"C:\Users\DEPMORA1\Documents\Projects\aktuarvereinigung"
DATA_PATH = rf"{MAIN_PATH}\data"
OUTPUT_PATH = rf"{MAIN_PATH}\output"

# Data
total_data = pd.read_csv(rf"{DATA_PATH}\insurance_claims.csv")

# %% Parameters

NUMBER_OF_RANDOM_STATES = 3
NUM_CLAIMS = len(total_data)
NUM_SPOT_CHECKS = 250

# %% Matplotlib Setting

# Matplotlib use in the backend
matplotlib.use("Agg")

# Matplotlib Sizes
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
plt.rc("font", size=BIGGER_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

# %% Status Quo

"""
At the very beginning we start to show why we would need any kind of prediction model anyway. We do that by first
showing how much money would be lost if no claim would be checked and how we the result improve if random
checks would be conducted
"""

spot_checks = np.random.choice([0, 1], size=NUM_CLAIMS, p=[(1-NUM_SPOT_CHECKS/NUM_CLAIMS), NUM_SPOT_CHECKS/NUM_CLAIMS])
y = total_data.loc[:, "fraud_reported"].replace({"Y": 1, "N": 0})
claim_amount_series = total_data.loc[:, "total_claim_amount"]
spot_checks_result = self_func.monetary_score_calc("no_checking", y, spot_checks, claim_amount_series)

# %% Missing values

"""
As one of the first things we take a look on how many observations are missing. Looking at the dataset we find
that question marks where used instead of NaNs. In order to give a picture of how much is missing, we replace
these question marks with NaNs
"""

total_data.replace("?", np.nan, inplace=True)
missing_data_per_column = total_data.apply(lambda x: (sum(x.isna()) / len(x)) * 100)
sorted_missing_pct = missing_data_per_column.sort_values(ascending=False)
fig, axs = plt.subplots(figsize=(20, 10))
sorted_missing_pct.plot(kind="bar", ax=axs)
path = rf"{OUTPUT_PATH}\missing_obs.png"
fig.savefig(path, bbox_inches="tight")
plt.close()

"""
Looking at the amount of missing data we drop the variable _39 and try to impute the others. For knowing what
kind of inputting would be appropriate we will have to take a look at the variable type.
"""

drop_columns = sorted_missing_pct[sorted_missing_pct == 100].index
total_data.drop(columns=list(drop_columns), inplace=True)

# %% Target Variable

"""
Looking at the count plot of the target variable below we can see that the majority of the data is "no fraud".
A finding that was to be expected. Though it has to be said that the imbalance does not look as bad as initially
feared. Given that would like to decrease the False Negative we can think about upsampling methods like SMOTE.
"""

num_of_target_values = total_data.loc[:, "fraud_reported"].value_counts()
imbalance_ratio = min(num_of_target_values) / max(num_of_target_values)
fig, axs = plt.subplots(figsize=(10, 10))
sns.countplot(total_data.loc[:, "fraud_reported"], ax=axs)
path = rf"{OUTPUT_PATH}\count_plot.png"
fig.savefig(path, bbox_inches="tight")
plt.close()

# %% Feature Engineering

"""
Before exploring the variables it would make sense to change format of some of them in beforehand
    - Age: Bucketing this variable would increase the amount of data in each bucket (buckets of 10 years)
    - Incident Date: Create Day of the Month/ Month/ Year information
    - Policy Bind Date: Create a variable to indicate how much time has passed since policy bind date
    - Incident Location: Stripping the house number of the road - though unlikely to have the same street
    - ZIP Code: Binned, given that the neighbourhood is much more important than the actual value
"""

# Age
min_age, max_age = total_data.loc[:, "age"].min(), total_data.loc[:, "age"].max()
bins_range = 10
bin_list = list(range(min_age, max_age+bins_range, bins_range))
label_list = [f"{x}_{y}" for (x, y) in zip(bin_list, bin_list[1:])]
total_data.loc[:, "age"] = pd.cut(total_data.loc[:, "age"], bins=bin_list, labels=label_list, include_lowest=True)
assert len(set(total_data.loc[:, "age"])) == len(label_list), "Not the right amount of categories"

for column in ["incident_date", "policy_bind_date"]:
    for name, date_series in zip(["year", "month", "day", "delta"],
                                 self_func.date_variable_creation(column, total_data)):
        total_data.loc[:, f"{column}_{name}"] = date_series
    total_data.drop(columns=column, inplace=True)
total_data.drop(columns="incident_date_year", inplace=True) # Since there is only one year

# Incident Location
series_incident_location = total_data.loc[:, "incident_location"]
replaced_inc_loc = series_incident_location.str.replace("\d+", "") # Removes all kind of numbers, but does the job
total_data.loc[:, "incident_location"] = replaced_inc_loc

# ZIP Code
zip_information = total_data.loc[:, "insured_zip"]
total_data.drop(columns="insured_zip", inplace=True)
total_data.loc[:, "high_level_zip"] = zip_information.astype(str).str[0:2]

# %% Column Type Divider

"""
Now we divide each variable into one variable category. This is done to generate appropriate plots for all variable
groups afterwards.
"""

feature_data = total_data.drop(columns="fraud_reported")
all_columns = feature_data.columns
cat_variables = [x for x in all_columns if self_func.categorical_check(x, feature_data)]
bin_variables = [x for x in all_columns if self_func.binary_check(x, feature_data)]
float_variables = [x for x in all_columns if self_func.float_check(x, feature_data)]
nom_ord_variables = feature_data.columns[feature_data.dtypes == object]

assert bool(set(cat_variables) & set(bin_variables) & set(float_variables)) is False, "Multiple Columns in diff Cat"
assert len(cat_variables) + len(bin_variables) + len(float_variables) == len(all_columns), "Not all Columns assigned"

df_num_variables = pd.DataFrame()
for i, (name, variable_list) in enumerate(zip(["Categorical", "Binary", "Floats"],
                                              [cat_variables, bin_variables, float_variables])):
    df_num_variables.loc[i, "Variable Type"] = name
    df_num_variables.loc[i, "Count"] = len(variable_list)

fig, axs = plt.subplots(figsize=(10, 10))
sns.barplot(x="Variable Type", y="Count", data=df_num_variables, ax=axs)
path = rf"{OUTPUT_PATH}\number_variables_of_category.png"
fig.savefig(path, bbox_inches="tight")
plt.close()

"""
We see that most variables are categorical, followed by floats. Furthermore, we do not have any binary variable.
It is to be said that for some float variables like zip-code, it is not clear yet whether they should
"""

# %% Variable Exploration

"""
[self_func.categorical_plots(x, total_data) for x in cat_variables]
[self_func.float_plots(x, total_data) for x in float_variables]
[self_func.nominal_ordinal_plots(x, total_data) for x in nom_ord_variables]
[self_func.float_outliers(x, total_data) for x in float_variables]
"""
"""
Insights:
    - ...
    - ...
    - ...
"""

# %% Variable Type Change

"""
Before moving on to the encoding, we have to think about what kind of encoding we would like to apply
and whether the same kind of encoding should be applied to every column. One common rule of thumb is to
apply dummy variables for categorical variables which have fewer than 5 categories. For all other
categories we apply TargetEncoding.

Furthermore, we have to decide in beforehand what we do with missing values for categorical values. One
possibility would be to build a classification model for those columns. Another approach would be to
treat the missing values as their own category. Given that the columns which contain missing values have
a fair bit of them, we will go for the latter approach.
"""

# For categorical variables it is decided that the missing values are treated as their own category
replaced_nan_values = feature_data.loc[:, bin_variables + cat_variables].astype(object).fillna("NaN")
feature_data.loc[:, bin_variables + cat_variables] = replaced_nan_values
assert not feature_data.loc[:, bin_variables].isna().any().any(), "Conversion did not work"

# Assign variables into their new category
cat2bin_variables = [x for x in cat_variables if self_func.less_than_six_categories(x, feature_data)]
bin_variables += cat2bin_variables
cat_variables = list(set(cat_variables) - set(bin_variables))

# %% Preprocessing Test

"""
Before we can start with the model building process, we have to transform some of the variables. The type of
preprocessing is dependent on the variable type

Categorical:
    - Less than, equal to 5 categories:
        - Creation of dummies
    - More than 5 categories:
        - Target Encoding
Float:
    - Standard-scaling

After all variables are transformed into a numeric format we apply the MICE algorithm in order to predict the
missing variables.
"""
# Float Variables
float_transformer = Pipeline(steps=[
    ("scaling", MinMaxScaler())
])
# Binary Variables
binary_transformer = Pipeline(steps=[
    ("one_hot_encoding", OneHotEncoder(sparse=False))
])
# Categorical Variables
categorical_transformer = Pipeline(steps=[
    ("target_encoding", TargetEncoder(handle_missing="return_nan")),
    ("scaling", MinMaxScaler())
])
# All together
variable_transformer = ColumnTransformer(
    transformers=[
        ("floats", float_transformer, float_variables),
        ("binary", binary_transformer, bin_variables),
        ("categorical", categorical_transformer, cat_variables),
    ]
)
# Imputing Pipeline
processing_pipeline = Pipeline(steps=[
    ("preprocesing", variable_transformer),
    ("imputing_missing", IterativeImputer())
])

processed_data = processing_pipeline.fit_transform(feature_data, y)
processed_data_columns_raw = self_func.get_feature_names(variable_transformer)
processed_data_columns = self_func.onehotencoder_namechange(processed_data_columns_raw, bin_variables)
assert not np.isnan(processed_data).any(), "Still Missing Data"

# %% Correlation Analysis

"""
In order to get a first idea which variable could be relevant in determining whether we face a fraud we look
at a correlation matrix which is generated through dummy encoded matrix
"""

df_target_binary = pd.get_dummies(feature_data)
df_target_binary.loc[:, "target"] = y
df_target_corr = df_target_binary.corr()
target_corr = df_target_corr.loc[:, ["target"]]

fig, axs = plt.subplots(ncols=2, figsize=(30, 10))
axs = axs.ravel()
for i, (bool_ascending, title) in enumerate(zip([True, False], ["Positive Correlation", "Negative Correlation"])):
    sorted_target_corr = target_corr.sort_values(by="target", ascending=bool_ascending)
    top_ten_target_corr = sorted_target_corr[:10]
    sns.heatmap(top_ten_target_corr, ax=axs[i])
    axs[i].set_yticklabels(axs[i].get_yticklabels(), rotation=45)
    axs[i].set_title(title, size=20)
    path = rf"{OUTPUT_PATH}\correlation_w_target.png"
    fig.savefig(path, bbox_inches="tight")
plt.close()

"""
Insights:
    -
    -
    -
"""

# %% First Model Try

"""
Before doing some serious hyper-parameter tuning we first start with trying out different models and see how each
of them is performing. Performance is measured by classical classification methods and, potentially even more
important, by monetary terms. Given that we have only 1000 observations, and face an imbalanced dataset,
we shuffle around the random state and conduct the evaluation not only once, but multiple times.
This would not be necessary if facing more observations.
"""

# Model initializing
rfc_simple = RandomForestClassifier(random_state=28, n_jobs=-1)
gb_simple = GradientBoostingClassifier(random_state=28)
gnb_simple = GaussianNB()

list_simple_results = [pd.DataFrame()] * 2
for rs in tqdm(range(NUMBER_OF_RANDOM_STATES)):
    for model_name, model in zip(["RandomForest", "GradientBoosting", "NaiveBayes"],
                                 [rfc_simple, gb_simple, gnb_simple]):

        list_temp = self_func.prediction_evaluation(feature_data, processed_data, y,
                                                    model, model_name, rs)
        list_simple_results = self_func.appending_temp(list_temp, list_simple_results)

df_metrics, df_money = [list_simple_results[i].rename_axis("method").reset_index()
                        for i in range(len(list_simple_results))]

# Separate number of checks given different scale on y-axis needed
df_money_wo_checks = df_money.query("method != 'num_unnecessary_checks'")
df_unnecessary_checks = df_money.query("method == 'num_unnecessary_checks'")

# Plotting the results
dict_plotting = {
    "model_comp_classical": df_metrics,
    "model_comp_money": df_money_wo_checks,
    "model_comp_checks": df_unnecessary_checks
}

[self_func.plot_boxplot(value, key) for (key, value) in dict_plotting.items()]

"""
We can see that the Naives Bayes results saves the most money and loses the least. This comes of course with a
price. Namely we have significantly more checks, which also come with a cost. Before deciding which model
to use, we use the model in the middle, the GradientBoosting
"""

# %% SMOTE Illustration

"""
Given that we have so few observations where we actually find fraud, it would be beneficial to artificially
increase the amount of observations where fraud is happening. For that we look at the upsampling
algorithm called SMOTE which is more sophisticated than simply duplicating or dropping the minority or
majority class. Before applying this algorithm we show two scatterplots and the results of that
"""

# Now we initialize the SMOTE algorithm
over_sample = SMOTE(sampling_strategy=0.5)
x_data = processed_data.copy()
y_data = y.copy()
sampled_x, sampled_y = over_sample.fit_resample(x_data, y_data)

# Plotting both versions
self_func.pca_scatter_plot(processed_data, y, "raw")
self_func.pca_scatter_plot(sampled_x, sampled_y, "smote_upsampled_50")

# %% Model Hyper-parameters

"""
From the bar charts we see that the xxx model should be favored. We therefore apply now all kind of hyper-parameters
and test them with a train test split.
"""

imbalance_base_ratio = round(imbalance_ratio, 2) + 0.05

random_grid = {
    "gradientboostingclassifier__learning_rate": [1/pow(10, x) for x in range(1, 3)],
    "gradientboostingclassifier__max_depth": np.linspace(3, 30, num=3, dtype=int),
    "gradientboostingclassifier__min_samples_leaf": np.linspace(1, 30, num=3, dtype=int),
    "gradientboostingclassifier__n_estimators": [1_000],
    "smote__sampling_strategy": np.linspace(imbalance_base_ratio, 1, num=3)
}

over_sampler = SMOTE(random_state=28)
gb_model = GradientBoostingClassifier(random_state=28)
gb_pipeline = imblearn.pipeline.make_pipeline(over_sampler, gb_model)
gb_gs_clf = GridSearchCV(gb_pipeline, random_grid, n_jobs=-1, scoring="f1")
gb_params_name = "gradientboosting_params"

list_main_frames = [pd.DataFrame()] * 4
for i, rs in enumerate(tqdm(range(NUMBER_OF_RANDOM_STATES))):

    list_temp = self_func.prediction_evaluation(feature_data, processed_data,
                                                 y, gb_gs_clf, gb_params_name,
                                                 rs, grid_search=True,
                                                 columns_list=processed_data_columns)
    list_main_frames = self_func.appending_temp(list_temp, list_main_frames)

# Unpacking filled dataframes
(df_gb_param_metrics, df_gb_param_money,
 df_best_params, df_feature_importance) = [list_main_frames[i].rename_axis("method").reset_index()
                                           for i in range(len(list_main_frames))]

# %% Plotting final results

"""
Now it is time to plot and compare our model performance. Furthermore, we have to save our hyperparameterized model
"""

# Reshaping the best parameter data frame
num_hyper_parameter = len(random_grid.keys())
list_number_rs = sorted([x for number_params in range(num_hyper_parameter) for x in range(NUMBER_OF_RANDOM_STATES)])
df_best_params.loc[:, "rs_experiment"] = list_number_rs
reshaped_best_params = pd.pivot_table(df_best_params, values="scores", index="rs_experiment", columns="method")

# Plot best parameters
fig, axs = plt.subplots(ncols=num_hyper_parameter, figsize=(num_hyper_parameter * 10, 10), sharey="all")
axs = axs.ravel()
for i, (series_keys, series_values)  in enumerate(random_grid.items()):
    sns.countplot(x=series_keys, data=reshaped_best_params, order=series_values, ax=axs[i])
    axs[i].set_title(series_keys.split("__")[1])
path = rf"{OUTPUT_PATH}\best_hyper_parameters_count_plot.png"
fig.savefig(path, bbox_inches="tight")
plt.close()

# Now parameterize the model with those hyper parameter with mode and save it as a pickle
hyper_model = self_func.train_best_model(gb_pipeline, reshaped_best_params, processed_data, y)
model_pkl_filename = f"{OUTPUT_PATH}\hyper_trained_model.pkl"
model_pkl = open(model_pkl_filename, "wb")
pickle.dump(hyper_model, model_pkl)
model_pkl.close()

# Plot model performance
dict_gb_plotting = {
    "model_comp_classical": df_gb_param_metrics,
    "model_comp_money": df_gb_param_money.query("method!='num_unnecessary_checks'"),
    "model_comp_checks": df_gb_param_money.query("method=='num_unnecessary_checks'")
}

for (simple_key, simple_value), (hyp_key, hyp_value) in zip(dict_plotting.items(),
                                                            dict_gb_plotting.items()):
    df_only_gb = simple_value.query("model_name=='GradientBoosting'")
    combined_data = pd.concat([df_only_gb, hyp_value], axis=0)
    plot_name = f"comparison_{simple_key}"
    self_func.plot_boxplot(combined_data, plot_name)
