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
from sklearn import metrics 
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

# Read data form dataset columns-[Temp (Â°C),Dew Point Temp (Â°C),Rel Hum (%),Wind Spd (km/h),Visibility (km),Stn Press (kPa),Work,Workout,Headache]
df = pd.read_csv("dataset_without_datetime.csv");

#check the datatypes of the data
print(df.dtypes)

#check for the missing data
print(df['Temp (°C)'].unique())
print(df['Dew Point Temp (°C)'].unique())
print(df['Rel Hum (%)'].unique())
print(df['Wind Spd (km/h)'].unique())
print(df['Visibility (km)'].unique())
print(df['Work'].unique())
print(df['Workout'].unique())
print(df['Headache'].unique())


#Visibility has nan of 3 rows
df=df.dropna()


#Features for predection, copy all columns(X) except Heacache col(Y)
X = df.drop('Headache',axis=1).copy()
X_val = df.drop('Headache',axis=1).values


#Feature to predict->Headache(Y)
Y=df['Headache'].values

clf_dts=[]
X_train_arr=[]
Y_train_arr=[]
X_test_arr=[]
Y_test_arr=[]
probs_arr=[]

# randomly duplicating examples from the minority class (has Headache) and adding them to the training dataset to create more balanced training dataset
# Naive random over-sampling
ros = RandomOverSampler(random_state=0)
# X_resampled, Y_resampled = ros.fit_resample(X_val, Y)
# Synthetic Minority Oversampling Technique
# X_resampled, Y_resampled = SMOTE().fit_resample(X_val, Y)
#  Adaptive Synthetic  Oversampling Technique
X_resampled, Y_resampled = ADASYN().fit_resample(X_val, Y)
from collections import Counter
print(sorted(Counter(Y_resampled).items()))
X_val = X_resampled
Y = Y_resampled

#The folds are made by preserving the percentage of samples for each class.
#Iterate for 3 times
skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X_val,Y):
    
    #get testing and training data
    X_train, X_test = X_val[train_index], X_val[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    #preserve the values in the arrays
    X_train_arr.append(X_train)
    Y_train_arr.append(Y_train)
    X_test_arr.append(X_test)
    Y_test_arr.append(Y_test)
    
    #Use Decision Tree and fit the training data to the model
    clf_dt = DecisionTreeClassifier(random_state=30)
    clf_dt = clf_dt.fit(X_train,Y_train)
    
    #preserve all the 
    clf_dts.append(clf_dt)
    
    #get the predections for the test data
    y_pred = clf_dt.predict(X_test)
    
    #plot the confusion matrix for eact fold and print the Accuracy.
    plot_confusion_matrix(clf_dt,X_test,Y_test,display_labels=["No Headache","Has Headache"])
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    #plt.figure(figsize=(50,50))
    # plot_tree(clf_dt,filled=True,rounded=True,class_names=["No Headache","Has Headache"],feature_names=X.columns)
    
    #Accuracy: 0.7938461538461539
    #Accuracy: 0.9507692307692308
    #Accuracy: 0.96


plot_tree(clf_dt,filled=True,rounded=True,class_names=["No Headache","Has Headache"],feature_names=X.columns)
#choosing 3rd item since accuracy has 96%
i=2
X_train=X_train_arr[i]
Y_train=Y_train_arr[i]
X_test=X_test_arr[i]
Y_test=Y_test_arr[i]

#cost complexity pruning for calculating alpha valuse
path = clf_dts[i].cost_complexity_pruning_path(X_train,Y_train)
ccp_alphas=path.ccp_alphas
#do not consider the alpha with highest value
ccp_alphas=ccp_alphas[:-1]

clf_dts=[]

#clculating alpha PART-1

#for each alpha build the decision tree
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    clf_dt = clf_dt.fit(X_train,Y_train)
    clf_dts.append(clf_dt)

#claculate the scores of each decision tree
train_scores=[clf_dt.score(X_train,Y_train) for clf_dt in clf_dts]
test_scores=[clf_dt.score(X_test,Y_test) for clf_dt in clf_dts]

#plot the accuracy across differenr alpha values
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.plot(ccp_alphas,train_scores,marker='o',label="train",drawstyle='steps-post')
ax.plot(ccp_alphas,test_scores,marker='o',label="test",drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values=[]

#clculating alpha PART-2

#calculate decision tree for different alphas
#run 5 fold cross validation
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train,Y_train,cv=5)
    alpha_loop_values.append([ccp_alpha,np.mean(scores),np.std(scores)])

#plot the graph of alphas with mean accuracy    
alpha_results=pd.DataFrame(alpha_loop_values,columns=['alpha','mean_accuracy','std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std',marker='o',linestyle='--')

#from the figure the best fit alpha is 0.0009

clf_dt = DecisionTreeClassifier(random_state=0,ccp_alpha=0.024)

clf_dt = clf_dt.fit(X_train,Y_train)

plot_confusion_matrix(clf_dt, X_test, Y_test,display_labels=["No Headache","Has Headache"])
plt.figure(figsize=(50,50))
plot_tree(clf_dt,filled=True,rounded=True,class_names=["No Headache","Has Headache"],feature_names=X.columns)


# calculate precision, recall, and average_precision_score. Plot ROC curve.
precision = precision_score(Y_test, clf_dt.predict(X_test), average='binary')
recall_score = recall_score(Y_test, clf_dt.predict(X_test), average='binary')
print(precision, recall_score)

average_precision = average_precision_score(Y_test, clf_dt.predict_proba(X_test).T[1])

disp = plot_precision_recall_curve(clf_dt, X_test, Y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))



    








