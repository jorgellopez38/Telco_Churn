import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import prepare


################################################################### baseline function ############################################################


def get_baseline(df):
    # code to generate baseline 
    df['baseline'] = df['churn_Yes'].value_counts().idxmax()
    (df['churn_Yes'] == df['baseline']).mean()
    # clean f string
    display(Markdown(f"Baseline: {(df['churn_Yes'] == df['baseline']).mean()*100:.2f}%"))


################################################################### model prep function ############################################################



def model_prep(df1,df2,df3):
    X_train = df1.drop(columns=['churn_Yes', 'contract_type', 'customer_id', 'internet_service_type'])
    y_train = df1.churn_Yes

    X_val = df2.drop(columns=['churn_Yes', 'contract_type', 'customer_id', 'internet_service_type'])
    y_val = df2.churn_Yes

    X_test = df3.drop(columns=['churn_Yes', 'contract_type', 'customer_id', 'internet_service_type'])
    y_test = df3.churn_Yes
    seed=42

    return X_train, X_val, X_test, y_train, y_val, y_test

################################################################### decision tree function ############################################################


def get_dt(X_train, X_val, y_train, y_val):
    '''get decision tree accuracy on train and validate data'''

    # create classifier object
    clf3 = DecisionTreeClassifier(max_depth=7, random_state=42)

    #fit model on training data
    clf3 = clf3.fit(X_train, y_train)

    # print result
    print(f"Accuracy of Decision Tree on train data is {clf3.score(X_train, y_train)}")
    print(f"Accuracy of Decision Tree on validate data is {clf3.score(X_val, y_val)}")

################################################################### random forest function ############################################################
    
    
def get_forest(X_train, X_val, y_train, y_val):
    '''get random forest accuracy on train and validate data'''
    
    rf3 = RandomForestClassifier(max_depth=8, random_state=42,
                            max_samples=0.5)

    rf3.fit(X_train, y_train)
    print(f"Accuracy of Random Forest on train data: {rf3.score(X_train, y_train)}") 
    print(f"Accuracy of Random Froest on validate: {rf3.score(X_val, y_val)}")


################################################################### logit function ############################################################
    
    
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


################################################################### knn function ############################################################


def get_knn(X_train, X_val, y_train, y_val):
    '''get KNN accuracy on train and validate data'''

    # create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=6, weights='uniform')
    knn.fit(X_train, y_train)

    # print results
    print(f"Accuracy of Logistic Regression on train is {knn.score(X_train, y_train)}")
    print(f"Accuracy of Logistic Regression on validate is {knn.score(X_val, y_val)}")


################################################################### top models function ############################################################


def get_top_models(X_train, y_train, X_validate, y_validate):
    '''This function gets all the top ML models and plots them together for a visual'''

    # best Random Forest
    best_rf = RandomForestClassifier(max_depth=8, random_state=42,
                                    max_samples=0.5)
    best_rf.fit(X_train, y_train)

    best_rf_train_score = best_rf.score(X_train, y_train)
    best_rf_validate_score = best_rf.score(X_validate, y_validate)

    # Best KNN
    best_knn = KNeighborsClassifier(n_neighbors=3)
    best_knn.fit(X_train, y_train)

    best_knn_train_score = best_knn.score(X_train, y_train)
    best_knn_validate = best_knn.score(X_validate, y_validate)

    # Best Logistic Regression
    best_lr = LogisticRegression(C=.1, random_state=42, 
                                intercept_scaling=1, solver='lbfgs')
    best_lr.fit(X_train, y_train)

    best_lr_train_score = best_lr.score(X_train, y_train)
    best_lr_validate_score = best_lr.score(X_validate, y_validate)

    # Best Decision Tree
    best_clf = DecisionTreeClassifier(max_depth=8, random_state=42)  
    best_clf.fit(X_train, y_train)

    best_clf_train_score = best_clf.score(X_train, y_train)
    best_clf_validate_score = best_clf.score(X_validate, y_validate)

    # lists with model names & score information
    best_model_name_list = ["KNN","Random_Forest","Logistic_Regression","Decision Tree"]
    best_model_train_scores_list = [best_knn_train_score,best_rf_train_score,best_lr_train_score,best_clf_train_score]
    best_model_validate_scores_list = [best_knn_validate,best_rf_validate_score,best_lr_validate_score,best_clf_validate_score]
    
    # new empty DataFrame
    best_scores_df = pd.DataFrame()

    # new columns using lists for data
    best_scores_df["Model"] = best_model_name_list
    best_scores_df["Train_Score"] = best_model_train_scores_list
    best_scores_df["Validate_Score"] = best_model_validate_scores_list

    # plot it
    plt.figure(figsize=(11, 8.5))
    ax = best_scores_df.plot.bar(rot=5)
    baseline_accuracy = .7346
    plt.axhline(baseline_accuracy , label="Baseline Accuracy", color='red')
    plt.xticks(np.arange(4), ['KNN', 'Random Forest','Logistic Regression', 'Decision Tree'])
    plt.ylabel('Scores')
    plt.title('Top Models')
    sns.set_theme(style="whitegrid")
    ax.annotate('Best Model',fontsize=12,color="Black",weight="bold", xy=(1, 1), 
                xytext=(.65, .9))
    mylabels = ['Baseline','Train', 'Validate']
    ax.legend(labels=mylabels,bbox_to_anchor=(1.02, 1), loc='upper left',borderaxespad=0)
    plt.show()


################################################################### test function ############################################################


def get_test(X_train, y_train, X_test, y_test):
    '''
    This function gets our best peforming model and runs it on our test data
    '''
    # random forest model was best
    rf3 = RandomForestClassifier(max_depth=8, random_state=42,
                            max_samples=0.5)

    #fit the model
    rf3.fit(X_train, y_train)
    rf3.score(X_test,y_test)
    
     # clean f string
    display(Markdown(f'### Random Forest Model'))
    display(Markdown(f'### Accuracy on Test {rf3.score(X_test,y_test)*100:.2f}%'))
    

################################################################### baseline vs test function ############################################################


def get_mvb(X_train, y_train, X_test, y_test, df):
    '''This function plots the test data and plot the baseline together for a final visual'''
    
    # Recalculating Best Peforming Model with new name
    best_model = RandomForestClassifier(max_depth=8, random_state=42,
                            max_samples=0.5) 
    best_model.fit(X_train, y_train)
    best_model.score(X_test,y_test)
    
    # Baseline
    df['baseline'] = df['churn_Yes'].value_counts().idxmax()
    plot_baseline = (df['churn_Yes'] == df['baseline']).mean()
    
    # Best Performing Model(Logistic Regression Combo{c=100,newton-cg}) Test Score: 
    best_test_score = best_model.score(X_test,y_test)  
    
    # Test Scores: Project Baseline vs Best Model
    plot_baseline, best_test_score
    
    # Temporary Dictionary Holding Baseline & Model Test Score
    best_model_plot={"Baseline":[plot_baseline], "Test":[best_test_score]}
    
    # Converting Temporary Dictionary to DataFrame
    best_model_plot = pd.DataFrame(best_model_plot)
    
    # Visualizing Both Baseline & Model Test Scores
    fig=sns.barplot(data= best_model_plot,palette="colorblind")
    plt.title("Baseline vs. Best Model")
    fig.set(ylabel='Scores')
    plt.show()