import pandas as pd
import numpy as np #to calculate mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs
from sklearn.tree import DecisionTreeClassifier # to build classification tree
from sklearn.tree import plot_tree #to draw the classification tree
from sklearn.model_selection import train_test_split #split data into Training and Testing datasets
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.metrics import confusion_matrix #to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix # to plot the confunsion matrix
from sklearn.model_selection import StratifiedKFold # Stratification K-fold
from sklearn.metrics import recall_score as RecallScore
from sklearn.metrics import average_precision_score as AveragePrecisionScore
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import precision_score

from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

class TrainTestDecisionTree:

    def __init__(self):
        self.clf_dts = []
        self.X_train_arr = []
        self.Y_train_arr = []
        self.X_test_arr = []
        self.Y_test_arr = []
        self.probs_arr = []
        self.X = None
        # features (columns)
        self.X_val = None
        # labels vector
        self.Y = None

        # final_decision_tree
        self.clf_dt = None

        self.X_test = None
        self.Y_test = None

    def read_and_clean_dataset(self):
        # Read data form dataset columns-[Temp (Â°C),Dew Point Temp (Â°C),Rel Hum (%),Wind Spd (km/h),Visibility (km),Stn Press (kPa),Work,Workout,Headache]
        df = pd.read_csv("dataset_without_datetime.csv");

        # check the datatypes of the data
        print(df.dtypes)

        # check for the missing data
        print(df['Temp (°C)'].unique())
        print(df['Dew Point Temp (°C)'].unique())
        print(df['Rel Hum (%)'].unique())
        print(df['Wind Spd (km/h)'].unique())
        print(df['Visibility (km)'].unique())
        print(df['Work'].unique())
        print(df['Workout'].unique())
        print(df['Headache'].unique())

        # Visibility has nan of 3 rows
        df = df.dropna()

        # Features for predection, copy all columns(X) except Heacache col(Y)
        self.X = df.drop('Headache', axis=1).copy()
        self.X_val = df.drop('Headache', axis=1).values

        # Feature to predict->Headache(Y)
        self.Y = df['Headache'].values
        return df

    def balance_dataset_using_oversampling(self):
        # randomly duplicating examples from the minority class (has Headache) and adding them to the training dataset to create more balanced training dataset
        # Naive random over-sampling
        ros = RandomOverSampler(random_state=0)
        # X_resampled, Y_resampled = ros.fit_resample(X_val, Y)
        # Synthetic Minority Oversampling Technique
        # X_resampled, Y_resampled = SMOTE().fit_resample(X_val, Y)
        #  Adaptive Synthetic  Oversampling Technique
        X_resampled, Y_resampled = ADASYN().fit_resample(self.X_val, self.Y)
        from collections import Counter
        print(sorted(Counter(Y_resampled).items()))
        self.X_val = X_resampled
        self.Y = Y_resampled

    def train_dataset_using_k_fold_stratification(self):
        # The folds are made by preserving the percentage of samples for each class.
        # Iterate for 3 times
        skf = StratifiedKFold(n_splits=3)
        for train_index, test_index in skf.split(self.X_val, self.Y):
            # get testing and training data
            X_train, X_test = self.X_val[train_index], self.X_val[test_index]
            Y_train, Y_test = self.Y[train_index], self.Y[test_index]

            # preserve the values in the arrays
            self.X_train_arr.append(X_train)
            self.Y_train_arr.append(Y_train)
            self.X_test_arr.append(X_test)
            self.Y_test_arr.append(Y_test)

            # Use Decision Tree and fit the training data to the model
            clf_dt = DecisionTreeClassifier(random_state=30)
            clf_dt = clf_dt.fit(X_train, Y_train)

            # preserve all the
            self.clf_dts.append(clf_dt)

            # get the predections for the test data
            y_pred = clf_dt.predict(X_test)

            # plot the confusion matrix for eact fold and print the Accuracy.
            plot_confusion_matrix(clf_dt, X_test, Y_test, display_labels=["No Headache", "Has Headache"])
            print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
            # plt.figure(figsize=(50,50))
            # plot_tree(clf_dt,filled=True,rounded=True,class_names=["No Headache","Has Headache"],feature_names=X.columns)

            # Accuracy: 0.7938461538461539
            # Accuracy: 0.9507692307692308
            # Accuracy: 0.96

        plot_tree(clf_dt, filled=True, rounded=True, class_names=["No Headache", "Has Headache"],
                  feature_names=self.X.columns)


    def prune_decision_tree(self):
        # choosing 3rd item since accuracy has 96%
        i = 2
        X_train = self.X_train_arr[i]
        Y_train = self.Y_train_arr[i]
        self.X_test = self.X_test_arr[i]
        self.Y_test = self.Y_test_arr[i]

        # cost complexity pruning for calculating alpha valuse
        path = self.clf_dts[i].cost_complexity_pruning_path(X_train, Y_train)
        ccp_alphas = path.ccp_alphas
        # do not consider the alpha with highest value
        ccp_alphas = ccp_alphas[:-1]

        clf_dts = []

        # clculating alpha PART-1

        # for each alpha build the decision tree
        for ccp_alpha in ccp_alphas:
            clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf_dt = clf_dt.fit(X_train, Y_train)
            clf_dts.append(clf_dt)

        # claculate the scores of each decision tree
        train_scores = [clf_dt.score(X_train, Y_train) for clf_dt in clf_dts]
        test_scores = [clf_dt.score(self.X_test, self.Y_test) for clf_dt in clf_dts]

        # plot the accuracy across differenr alpha values
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle='steps-post')
        ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle='steps-post')
        ax.legend()
        plt.show()

        alpha_loop_values = []

        # clculating alpha PART-2

        # calculate decision tree for different alphas
        # run 5 fold cross validation
        for ccp_alpha in ccp_alphas:
            clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            scores = cross_val_score(clf_dt, X_train, Y_train, cv=5)
            alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

        # plot the graph of alphas with mean accuracy
        alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
        alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')

        # from the figure the best fit alpha is 0.0009

        self.clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=0.024)

        self.clf_dt = self.clf_dt.fit(X_train, Y_train)

        plot_confusion_matrix(self.clf_dt, self.X_test, self.Y_test, display_labels=["No Headache", "Has Headache"])
        plt.figure(figsize=(50, 50))
        plot_tree(self.clf_dt, filled=True, rounded=True, class_names=["No Headache", "Has Headache"],
                  feature_names=self.X.columns)

    def model_evaluation(self):
        # calculate precision, recall, and average_precision_score. Plot ROC curve.
        precision = precision_score(self.Y_test, self.clf_dt.predict(self.X_test), average='binary')

        recall_score = RecallScore(self.Y_test, self.clf_dt.predict(self.X_test), average='binary')
        print(precision, recall_score)

        average_precision = AveragePrecisionScore(self.Y_test, self.clf_dt.predict_proba(self.X_test).T[1])

        disp = plot_precision_recall_curve(self.clf_dt, self.X_test, self.Y_test)
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))


if __name__ == "__main__":
    trainTestDecisionTree = TrainTestDecisionTree()
    trainTestDecisionTree.read_and_clean_dataset()
    trainTestDecisionTree.balance_dataset_using_oversampling()
    trainTestDecisionTree.train_dataset_using_k_fold_stratification()
    trainTestDecisionTree.prune_decision_tree()
    trainTestDecisionTree.model_evaluation()
    print("ok")