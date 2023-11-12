# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Other umports
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import random
from scipy.stats import lognorm, loguniform, randint


from sklearn.preprocessing import StandardScaler


# Classifiers and regressors
from sklearn.dummy import DummyClassifier, DummyRegressor

# train test split and cross validation
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from lightgbm.sklearn import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import RandomizedSearchCV

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve


# functions

def bmi_feat(df):
    ''' creates the bmi and the associated risk colum
    BMI < 18.5 - Underwieght
    BMI between 18.5- 24.99 - Normal
    BMI between 25- 29.99 - Overweight
    BMI between 30-34.99 - Obese- grade 1
    BMI between 35- 39.99 - Obese- grade 2
    BMI >= 40.00 - Obese- grade 3 
    '''
    
    
    bmi_ranges = [0, 18.5, 25, 30, 35, 40, float('inf')]
    bmi_levels = ['Underweight', 'Normal', 'Overweight', 'Obese-lvl1','Obese-lvl2', 'Obese-lvl3']
    
    df['bmi'] = df['weight(kg)'] / ((df['height(cm)']*0.01)**2)
    df['bmi_category'] = pd.cut(df['bmi'], bins = bmi_ranges, labels=bmi_levels, right=False)
    
    return df

def obesity_risk(df):
    ''' creates col for obesity risk based on waistline

        Waist circumference and obesity
        For men-
        below 94cm (37in) - low risk
        94–102cm (37-40in) - high risk
        more than 102cm (40in) - very high
        For women-
        below 80cm (31.5in) - low risk
        80–88cm (31.5-34.6in) - high risk
        more than 88cm (34.6in) - very high
    '''
        
    female = df['male_probability'] < 0.5
    male = df['male_probability'] >= 0.5
    
    male_waist_ranges = [0,94,102,float('inf')]
    female_waist_ranges = [0,80,88,float('inf')]
    
    waist_levels = ['low risk', 'high risk', 'very high risk']
    
    df['obesity_risk'] = np.where(female, pd.cut(df['waist(cm)'], bins=female_waist_ranges, labels=waist_levels), pd.cut(df['waist(cm)'], bins=male_waist_ranges, labels=waist_levels))
    
    return df

def age_risk(df):
    ''' risk based on age
        
        Age and health risk
        Age < 45 - lower risk
        Age >= 45 years - high risk 
    '''
    # for now computing with the solo gender info as men
    age_ranges = [0,45,float('inf')]
    age_levels = ['low risk', 'high risk']
    
    df['age_risk'] = pd.cut(df['age'], bins = age_ranges, labels=age_levels, right=False)
    
    return df


def bp_risk(df):
    '''
    calculates the bp risk
    
    Systolic BP < 120 mmHg or Diastolic BP < 80 mmHg - normal
    Systolic BP between 120 mmHg- 129 mmHg or Diastolic BP between 80-89 mmHg- elevated
    Systolic BP between 130 mmHg - 139 mmHg or Diastolic BP between 90-99 mmHg- stage 1 high BP
    Systolic BP between 140 mmHg - 179 mmHg or Diastolic BP between 100-119 mmHg- stage 2 high BP
    Systolic BP >= 180 mmHg or Diastolic BP >= 120 mmHg- stage 3 high BP (emergency)
        '''
    
    bp_ranges = [0,120,130,140,180,float('inf')]
    bp_levels = ['normal', 'elevated', 'high_lvl1', 'high_lvl2', 'high_lvl3']
    
    df['bp_risk'] = pd.cut(df['systolic'], bins = bp_ranges, labels=bp_levels, right=False)
    
    return df
    
    
def hdl_risk(df):
    '''
    Less then 40- High risk
    Between 40-60 - Normal
    Greater of equals 60- Low risk
    '''
    hdl_ranges = [0,40,60,float('inf')]
    hdl_levels = ['high risk', 'normal', 'low risk']
    
    df['hdl_risk'] = pd.cut(df['HDL'], bins = hdl_ranges, labels=hdl_levels, right=False)
    
    return df

def ldl_risk(df):
    '''
    Less than 100mg/dL- Optimal (best for your health)
    100-129mg/dL- Near optimal
    130-159 mg/dL- Borderline high
    160-189 mg/dL- High
    190 mg/dL and above- Very High
    '''
    ldl_ranges = [0,100,130,160,190,float('inf')]
    ldl_levels = ['optimal', 'normal', 'high_lvl1', 'high_lvl2', 'high_lvl3']
    
    df['ldl_risk'] = pd.cut(df['LDL'], bins = ldl_ranges, labels=ldl_levels, right=False)
    
    return df   

def tg_risk(df):
    '''
    Less than 150 - normal
    150- 199- moderate risk
    200-499- high risk
    500+ - very high risk 
    '''
    tg_ranges = [0,150,200,500,float('inf')]
    tg_levels = ['normal', 'moderate_risk', 'high_risk', 'very_high_risk']
    
    df['tg_risk'] = pd.cut(df['triglyceride'], bins = tg_ranges, labels=tg_levels, right=False)
    
    return df      

def anemic_risk(df):
    '''
    hemoglobin
    Less than 12
    '''
    
    hg_ranges = [0,12,float('inf')]
    hg_levels = ['high risk', 'low risk']
    
    df['anemic_risk'] = pd.cut(df['hemoglobin'], bins = hg_ranges, labels=hg_levels, right=False)
    
    return df   


def creatinine_range(df):
    '''
    For adult men - 0.74 to 1.35 mg/dL (65.4 to 119.3 micromoles/L)
    For adult women - 0.59 to 1.04 mg/dL (52.2 to 91.9 micromoles/L) 
    '''
    
    female = df['male_probability'] < 0.5
    male = df['male_probability'] >= 0.5
    
    female_cr_ranges = [0,0.60, 1.04, float('inf')]
    male_cr_ranges = [0,0.75, 1.35, float('inf')]
    cr_levels = ['low','normal', 'high']
    
    df['creatinine_cat'] = np.where(female, pd.cut(df['serum creatinine'], bins=female_cr_ranges, labels=cr_levels), pd.cut(df['serum creatinine'], bins=male_cr_ranges, labels=cr_levels))
    
    return df    


def gtp_range(df):
    '''
    normal range for adults is 5 to 40 U/L
    values above 40 are considered risky for liver disease 
    '''

    gtp_ranges = [0,5, 40, float('inf')]
    gtp_levels = ['low','normal', 'high']
    
    df['gtp_cat'] = pd.cut(df['Gtp'], bins = gtp_ranges, labels=gtp_levels, right=False)
    
    return df  


def ast_range(df):
    '''
    The normal range of an SGOT test is generally between 8 and 45 units per liter of serum
    Higher results indicate risk for liver disease    
    '''
    ast_ranges = [0,8, 45, float('inf')]
    ast_levels = ['low','normal', 'high']
    
    df['ast_cat'] = pd.cut(df['AST'], bins = ast_ranges, labels=ast_levels, right=False)
    
    return df  

def alt_range(df):
    '''
    The normal range of an alt test is generally between 29 to 33 
    international units per liter (IU/L) for males and 19 to 25 IU/L for females
    '''
    alt_ranges = [0,29, 33, float('inf')]
    alt_levels = ['low','normal', 'high']
    
    df['alt_cat'] = pd.cut(df['ALT'], bins = alt_ranges, labels=alt_levels, right=False)
    
    return df  