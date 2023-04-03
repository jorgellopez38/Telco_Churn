import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from IPython.display import display, Markdown
import pandas as pd
import prepare

################################################################### prepare data for modeling ############################################################

def get_baseline(df):
    # code to generate baseline 
    df['baseline'] = df['churn_Yes'].value_counts().idxmax()
    (df['churn_Yes'] == df['baseline']).mean()
    # clean f string
    print(f"Baseline: {(df['churn_Yes'] == df['baseline']).mean()*100:.2f}%")


def model_prep(df1,df2,df3):
    X_train = df1.drop(columns=['churn_Yes', 'contract_type', 'customer_id', 'internet_service_type'])
    y_train = df1.churn_Yes

    X_val = df2.drop(columns=['churn_Yes', 'contract_type', 'customer_id', 'internet_service_type'])
    y_val = df2.churn_Yes

    X_test = df3.drop(columns=['churn_Yes', 'contract_type', 'customer_id', 'internet_service_type'])
    y_test = df3.churn_Yes
    seed=42

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_dt(X_train, X_val, y_train, y_val):
    '''get decision tree accuracy on train and validate data'''

    # create classifier object
    clf3 = DecisionTreeClassifier(max_depth=8, random_state=42)

    #fit model on training data
    clf3 = clf3.fit(X_train, y_train)

    # print result
    print(f"Accuracy of Decision Tree on train data is {clf3.score(X_train, y_train)}")
    print(f"Accuracy of Decision Tree on validate data is {clf3.score(X_val, y_val)}")
    
    
def get_forest(X_train, X_val, y_train, y_val):
    '''get random forest accuracy on train and validate data'''
    
    rf3 = RandomForestClassifier(max_depth=8, random_state=42,
                            max_samples=0.5)

    rf3.fit(X_train, y_train)
    print(f"Accuracy of Random Forest on train data: {rf3.score(X_train, y_train)}") 
    print(f"Accuracy of Random Froest on validate: {rf3.score(X_val, y_val)}")
    
    
def get_reg(X_train, X_val, y_train, y_val):
    '''get logistic regression accuracy on train and validate data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(C=.1, random_state=42, 
                               intercept_scaling=1, solver='lbfgs')
    # fit the model
    logit.fit(X_train, y_train)

    # print result
    print(f"Accuracy of Logistic Regression on train is {logit.score(X_train, y_train)}")
    print(f"Accuracy of Logistic Regression on validate is {logit.score(X_val, y_val)}")

def get_knn(X_train, X_val, y_train, y_val):
    '''get KNN accuracy on train and validate data'''

    # create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, y_train)

    # print results
    print(f"Accuracy of Logistic Regression on train is {knn.score(X_train, y_train)}")
    print(f"Accuracy of Logistic Regression on validate is {knn.score(X_val, y_val)}")


def get_test(X_train, y_train, X_test, y_test):
    '''
    This function gets our best peforming model and runs it on our test data
    '''
    # random forest model was best
    logit2 = LogisticRegression(C=.1, random_state=42, 
                           intercept_scaling=1, solver='lbfgs')

    #fit the model
    logit2.fit(X_train, y_train)
    logit2.score(X_test,y_test)
    
     # clean f string
    display(Markdown(f'### Logistic Regression Model'))
    display(Markdown(f'### Accuracy on Test {logit2.score(X_test,y_test)*100:.2f}%'))