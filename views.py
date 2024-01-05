import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import plotly.express as pl
import matplotlib
matplotlib.use('agg')


def run_eda():
    risk_df = pd.read_csv('files/credit_risk_preprocessed.csv')

    feature_corelation_df = risk_df[
        ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
         'cb_person_cred_hist_length', 'person_home_ownership_cat', 'loan_intent_cat']]
    feature_corelation_df.corr()

    # Plot for the correlation matrix

    corelation_plot, axis = plt.subplots()
    corelation_plot.set_size_inches(8, 8)
    sns.heatmap(feature_corelation_df.corr(), vmax=.8, square=True, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix', fontsize=20)

    plt.savefig("static/eda_images/credit_risk_correlation_matrix.png")


    age_count = risk_df['person_age'].value_counts().values
    age = risk_df['person_age'].value_counts().index
    fig = plt.figure(figsize=(8, 8))
    plt.bar(age, age_count)
    plt.title("Age Histogram", fontsize=20)
    plt.xlabel("Age", fontsize=15)
    plt.ylabel("Total Count", fontsize=15)
    # plt.show()
    plt.savefig("static/eda_images/credit_risk_age_histogram.png")

    plt.figure(figsize=(8, 8))
    sns.countplot(x=risk_df['loan_intent'], palette="Spectral")
    plt.title('Plot on Loan Intent Counts', fontsize=18)
    plt.xlabel('Loan Intent Types', fontsize=15)
    plt.ylabel('Total Count', fontsize=15)
    # plt.show()
    plt.savefig("static/eda_images/credit_risk_loan_intent.png")


    income_range_counts = risk_df.income_range.value_counts()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_title("Income Range of Users")
    ax.pie(income_range_counts.values, labels=income_range_counts.index, colors=sns.color_palette('cool'),
           autopct='%.0f%%')
    # fig.show()
    fig.savefig("static/eda_images/credit_risk_income_range.png")


def get_metrics(y_test, predicted):
    accuracy = accuracy_score(y_test, predicted)  # Accuracy\
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predicted, average='binary',
                                                                     zero_division=0)

    print(f'Accuracy of the model: {accuracy}')
    print(f'Precision of the model: {precision}')
    print(f'Recall of the model: {recall}')
    print(f'F1 score of the model: {f1_score}')

    metrics = {
        'accuracy' : accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return metrics


# Plotting Roc Curve
def plot_roc_curve(y_test, y_pred, model):
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Plot the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    model_cap = model.upper().replace('_', ' ')
    plt.title(f'Receiver Operating Characteristic (ROC) for Model {model_cap}')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(f"static/model_images/{model}_roc.png")

def plot_confusion_matrix(confusion_matrix, model):
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted loan_status')
    plt.ylabel('True loan_status')
    model_cap = model.upper().replace('_', ' ')
    plt.title(f'Confusion Matrix - {model_cap}')
    # plt.show()
    plt.savefig(f"static/model_images/{model}_confusion.png")

def save_model(model,filename):
    pickle.dump(model, open(filename, 'wb'))

def load_model(model,filename):
    return pickle.load(open(filename, 'rb'))


# Function for logistic regression for the dataset
def LogisticRegression_sklearn(X_train, Y_train, X_test):
    logistic_regression_model = LogisticRegression()

    # Fitting the train dataset
    logistic_regression_model.fit(X_train, Y_train)

    # Saving the model in a pickle file
    save_model(logistic_regression_model, 'static/models/logistic_regression.pkl')
    Y_pred = logistic_regression_model.predict(X_test)
    return Y_pred


# Function to implment Support vector machine algorithm for the dataset
def SVM_sklearn(X_train, Y_train, X_test):
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, Y_train)
    save_model(svm_model, 'static/models/support_vector_machine.pkl')
    Y_pred = svm_model.predict(X_test)
    return Y_pred


# Function to implment random forest algorithm for the dataset
def random_forest_sklearn(X_train, Y_train, X_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, Y_train)
    save_model(rf_model, 'static/models/random_forest.pkl')
    Y_pred = rf_model.predict(X_test)
    return Y_pred


# Function to implment decision trees algorithm for the dataset
def decision_tree_sklearn(X_train, Y_train, X_test):
    dt_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    dt_model.fit(X_train, Y_train)
    save_model(dt_model, 'static/models/decision_trees.pkl')
    Y_pred = dt_model.predict(X_test)
    return Y_pred

# Function to implment XGboost algorithm for the dataset
def xgboost_sklearn(X_train, Y_train, X_test):
    xgboost_model = xgb.XGBClassifier(objective='binary:logistic', seed=42)
    xgboost_model.fit(X_train, Y_train)
    save_model(xgboost_model, 'static/models/xgboost.pkl')
    Y_pred = xgboost_model.predict(X_test)
    return Y_pred

# Function to implment Naive Bayes algorithm for the dataset
def naive_bayes_sklearn(X_train, Y_train, X_test):
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, Y_train)
    save_model(naive_bayes_model, 'static/models/naive_bayes.pkl')
    Y_pred = naive_bayes_model.predict(X_test)
    return Y_pred

# Function to implment gradient boosting machines algorithm for the dataset
def gbm_sklearn(X_train, Y_train, X_test):
    gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
    gbm_model.fit(X_train, Y_train)
    save_model(gbm_model, 'static/models/gradient_boosting_machines.pkl')
    Y_pred = gbm_model.predict(X_test)
    return Y_pred



def run_models(model):
    risk_df = pd.read_csv('files/credit_risk_preprocessed.csv')

    risk_df.drop(columns=risk_df.columns[0], axis=1, inplace=True)
    X = risk_df[['age_range_cat', 'income_range_cat', 'person_home_ownership_cat', 'person_emp_length_normalized',
                 'loan_intent_cat', 'loan_grade_cat', 'loan_amount_range_cat', 'loan_int_rate_normalized',
                 'loan_percent_income', 'cb_person_default_on_file_cat', 'cb_person_cred_hist_length_normalized']]

    Y = risk_df[['loan_status']]

    # Splitting the dataset into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()

    Y_train = Y_train.ravel()

    if model == 'logistic_regression':
        logistic_regression_y_pred = LogisticRegression_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, logistic_regression_y_pred)

        cm_lr = confusion_matrix(Y_test, logistic_regression_y_pred)
        plot_confusion_matrix(cm_lr, model)

        plot_roc_curve(Y_test, logistic_regression_y_pred, model)

        return metrics

    elif model == 'support_vector_machine':
        svm_y_pred = SVM_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, svm_y_pred)

        cm_svm = confusion_matrix(Y_test, svm_y_pred)
        plot_confusion_matrix(cm_svm, model)

        plot_roc_curve(Y_test, svm_y_pred, model)

        return metrics

    elif model == 'random_forest':
        rf_y_pred = random_forest_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, rf_y_pred)

        cm_rf = confusion_matrix(Y_test, rf_y_pred)
        plot_confusion_matrix(cm_rf, model)

        plot_roc_curve(Y_test, rf_y_pred, model)

        return metrics

    elif model == 'decision_trees':
        decision_tree_y_pred = decision_tree_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, decision_tree_y_pred)

        cm_decision_tree = confusion_matrix(Y_test, decision_tree_y_pred)
        plot_confusion_matrix(cm_decision_tree, model)

        plot_roc_curve(Y_test, decision_tree_y_pred, model)

        return metrics

    elif model == 'xgboost':
        xgboost_y_pred = xgboost_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, xgboost_y_pred)

        cm_xgbooxt = confusion_matrix(Y_test, xgboost_y_pred)
        plot_confusion_matrix(cm_xgbooxt, model)

        plot_roc_curve(Y_test, xgboost_y_pred, model)

        return metrics

    elif model == 'naive_bayes':
        nb_y_pred = naive_bayes_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, nb_y_pred)

        cm_bayes = confusion_matrix(Y_test, nb_y_pred)
        plot_confusion_matrix(cm_bayes, model)

        plot_roc_curve(Y_test, nb_y_pred, model)

        return metrics

    elif model == 'gradient_boosting_machines':
        gbm_y_pred = gbm_sklearn(X_train, Y_train, X_test)
        metrics = get_metrics(Y_test, gbm_y_pred)

        cm_gbm = confusion_matrix(Y_test, gbm_y_pred)
        plot_confusion_matrix(cm_gbm, model)

        plot_roc_curve(Y_test, gbm_y_pred, model)

        return metrics


def get_predictions(input_data, model):
    if model == 'logistic_regression':
        prediction_model = pickle.load(open(f'static/models/{model}.pkl', 'rb'))
        prediction = prediction_model.predict([input_data])
        print(prediction[0])

        return "APPROVED" if prediction[0] == 1 else "REJECTED"

    elif model == 'random_forest':
        prediction_model = pickle.load(open(f'static/models/{model}.pkl', 'rb'))
        prediction = prediction_model.predict([input_data])
        print(prediction[0])

        return "APPROVED" if prediction[0] == 1 else "REJECTED"

    # elif model == 'support_vector_machine':
    #     svm_y_pred = SVM_sklearn(X_train, Y_train, X_test)
    #     metrics = get_metrics(Y_test, svm_y_pred)
    #
    #     cm_svm = confusion_matrix(Y_test, svm_y_pred)
    #     plot_confusion_matrix(cm_svm, model)
    #
    #     plot_roc_curve(Y_test, svm_y_pred, model)
    #
    #     return metrics
    #
    # elif model == 'random_forest':
    #     rf_y_pred = random_forest_sklearn(X_train, Y_train, X_test)
    #     metrics = get_metrics(Y_test, rf_y_pred)
    #
    #     cm_rf = confusion_matrix(Y_test, rf_y_pred)
    #     plot_confusion_matrix(cm_rf, model)
    #
    #     plot_roc_curve(Y_test, rf_y_pred, model)
    #
    #     return metrics
    #
    # elif model == 'decision_trees':
    #     decision_tree_y_pred = decision_tree_sklearn(X_train, Y_train, X_test)
    #     metrics = get_metrics(Y_test, decision_tree_y_pred)
    #
    #     cm_decision_tree = confusion_matrix(Y_test, decision_tree_y_pred)
    #     plot_confusion_matrix(cm_decision_tree, model)
    #
    #     plot_roc_curve(Y_test, decision_tree_y_pred, model)
    #
    #     return metrics
    #
    # elif model == 'xgboost':
    #     xgboost_y_pred = xgboost_sklearn(X_train, Y_train, X_test)
    #     metrics = get_metrics(Y_test, xgboost_y_pred)
    #
    #     cm_xgbooxt = confusion_matrix(Y_test, xgboost_y_pred)
    #     plot_confusion_matrix(cm_xgbooxt, model)
    #
    #     plot_roc_curve(Y_test, xgboost_y_pred, model)
    #
    #     return metrics
    #
    # elif model == 'naive_bayes':
    #     nb_y_pred = naive_bayes_sklearn(X_train, Y_train, X_test)
    #     metrics = get_metrics(Y_test, nb_y_pred)
    #
    #     cm_bayes = confusion_matrix(Y_test, nb_y_pred)
    #     plot_confusion_matrix(cm_bayes, model)
    #
    #     plot_roc_curve(Y_test, nb_y_pred, model)
    #
    #     return metrics
    #
    # elif model == 'gradient_boosting_machines':
    #     gbm_y_pred = gbm_sklearn(X_train, Y_train, X_test)
    #     metrics = get_metrics(Y_test, gbm_y_pred)
    #
    #     cm_gbm = confusion_matrix(Y_test, gbm_y_pred)
    #     plot_confusion_matrix(cm_gbm, model)
    #
    #     plot_roc_curve(Y_test, gbm_y_pred, model)
    #
    #     return metrics
    #
    #
    #
