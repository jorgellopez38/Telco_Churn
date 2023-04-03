import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
import prepare


#################################################### Functions ####################################################


def get_tech(df):
    ''' this funtion takes in a dataframe and returns a plot with tech support and churning'''
    # using a countplot to give a visual 
    sns.set_theme(style="whitegrid")
    sns.countplot(data=df, x="churn_Yes", hue="tech_support_Yes")
    #labels and legend
    plt.title('Churn Vs Tech Support')
    plt.xlabel('Did They Churn?')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.ylabel('# of Customers')
    title = 'Tech Support'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()
    

def question_hypothesis_test1(df):
    ''' this funtion takes in a dataframe and returns a chi square test'''
    # observed data for chi-square test
    observed1 = pd.crosstab(df.tech_support_Yes, df.churn_Yes)
    observed1
    # do not forget alpha
    alpha = .05

    chi2, p, degf, expected = stats.chi2_contingency(observed1)
    # clean f strings
    print(f"chi^2: {chi2}")
    print(f"p value: {p}")
    

def get_contracts(df):
    ''' this funtion takes in a dataframe and returns visual for contract type and churn'''
    # trusty count plot for visual
    sns.set_theme(style="whitegrid")
    sns.countplot(data=df, x="churn_Yes", hue="contract_type")
    #labels and legend
    plt.title('Churn Vs Types of Contracts')
    plt.xlabel('Churn')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.ylabel('# of Customers')
    title = 'Contract Type'
    plt.legend(title=title)
    plt.show()


def question_hypothesis_test2(df):
    ''' this funtion takes in a dataframe and returns a chi square test'''
    # observed data for chi-square test
    observed2 = pd.crosstab(df.contract_type, df.churn_Yes)
    observed2
    # do not forget alpha
    alpha = .05

    chi2, p, degf, expected = stats.chi2_contingency(observed2)
    # clean f strings
    print(f"chi^2: {chi2}")
    print(f"p value: {p}")
    

def get_seniors(df):
    ''' this funtion takes in a dataframe and returns a plot with senior citizens and churning'''
    sns.set_theme(style="whitegrid")
    sns.countplot(data=df, x="churn_Yes", hue="senior_citizen")
    #labels and legend
    plt.title('Churn Vs Senior Citizens')
    plt.xlabel('Churn')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.ylabel('# of Customers')
    title = 'Senior Citizen'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()
    
    
def question_hypothesis_test3(df):
    ''' this funtion takes in a dataframe and returns a chi square test'''
    # trusted chi-square test
    observed3 = pd.crosstab(df.senior_citizen, df.churn_Yes)
    observed3
    # do not forget alpha
    alpha = .05

    chi2, p, degf, expected = stats.chi2_contingency(observed3)
    # clean f strings
    print(f"chi^2: {chi2}")
    print(f"p value: {p}")
    

def get_paper(df):
    ''' this funtion takes in a dataframe and returns a plot with paperless billing and churning'''
    # count plot
    sns.set_theme(style="whitegrid")
    sns.countplot(data=df, x="churn_Yes", hue="paperless_billing_Yes")
    # title and legend
    plt.title('Churn Vs Paperless Billing')
    plt.xlabel('Churn')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.ylabel('# of Customers')
    title = 'Paperless Billing'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()
    
    
def question_hypothesis_test4(df):
    ''' this funtion takes in a dataframe and returns a chi square test'''
    observed4 = pd.crosstab(df.paperless_billing_Yes, df.churn_Yes)
    observed4
    # do not forget alpha
    alpha = .05

    chi2, p, degf, expected = stats.chi2_contingency(observed4)
    # clean f strings
    print(f"chi^2: {chi2}")
    print(f"p value: {p}")
    
