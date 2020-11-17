import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.impute import SimpleImputer
from scipy.stats import iqr

import holidays

class TimeHandler(TransformerMixin,BaseEstimator):
    """ Splits the time into hour, minute
        Creates features such as Period, Military Time
    """
    def __init__(self,time_cols_names:list,return_whole_df:bool,drop_original_col = True):
        self.return_whole_df = return_whole_df
        self.cols = []
        self.time_cols_names = time_cols_names
        self.added_cols_suffix = ["Hour","Minute","MilitaryTime","PeriodName","PeriodNum"]
        self.added_cols = []
        self.drop_original_col = drop_original_col

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        for time_col_name in self.time_cols_names:
            if self.drop_original_col:
                self.cols.remove(time_col_name)
            self.added_cols.extend([time_col_name + "_" + i for i in self.added_cols_suffix])
        return self


    def get_time_period_name(self,x):
        periods_name = ["Morning","Afternoon","Evening","Night"]
        
        # Mornining : 0500 to 1200
        # Afternoon : 1201 to 1700
        # Evening : 1701 to 2000
        # Night : 2001 : 0459

        if x >= 500 and x <= 1200:
            return periods_name[0]
        elif x >= 1201 and x <= 1700:
            return periods_name[1]
        elif x >= 1701 and x <= 2000:
            return periods_name[2]
        return periods_name[3]
    
    def get_time_period_num(self,x):
        
        periods_num = [1,2,3,4]
        # Mornining : 0500 to 1200
        # Afternoon : 1201 to 1700
        # Evening : 1701 to 2000
        # Night : 2001 : 0459

        if x >= 500 and x <= 1200:
            return periods_num[0]
        elif x >= 1201 and x <= 1700:
            return periods_num[1]
        elif x >= 1701 and x <= 2000:
            return periods_num[2]
        return periods_num[3]

    def transform(self,df,*_):
        copy = df.copy()
        for time_col_name in self.time_cols_names:
            copy[time_col_name] = pd.to_datetime(copy[time_col_name],format="%H:%M")

            copy[time_col_name+"_"+"Hour"] = copy[time_col_name].dt.hour
            copy[time_col_name+"_"+"Minute"] = copy[time_col_name].dt.minute
            # Military Time
            copy[time_col_name+"_"+"MilitaryTime"] = (copy[time_col_name+"_"+"Hour"].astype(str) + copy[time_col_name+"_"+"Minute"].astype(str)).astype(int)

            # Time Period
            copy[time_col_name+"_"+"PeriodName"] = copy[time_col_name+"_"+"Time"].apply(self.get_time_period_name)
            
            copy[time_col_name+"_"+"PeriodNum"] = copy[time_col_name+"_"+"Time"].apply(self.get_time_period_num)

         


        if self.return_whole_df:
            return copy[self.cols+self.added_cols]

        if self.drop_original_col:
            return copy[self.added_cols]

        return copy[self.time_cols_names + self.added_cols]

    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols
        return self.added_cols


class DateHandler(TransformerMixin,BaseEstimator):
    """ Splits the date into day, month, year
        Creates features such as day_name, is_weekend
    """
    def __init__(self,date_cols_names:list,return_whole_df:bool,drop_original_col = True):
        self.return_whole_df = return_whole_df
        self.cols = []
        self.date_cols_names = date_cols_names
        self.added_cols_suffix = ["Day","Month","Year","Day_Name","Day_Num","is_weekend","Close_To_Month_Start_End","is_holiday","Quater"]
        self.added_cols = []
        self.drop_original_col = drop_original_col


    def fit(self,df,y=None):
        self.cols = list(df.columns)
        for date_col_name in self.date_cols_names:
            if self.drop_original_col:
                self.cols.remove(date_col_name)
            self.added_cols.extend([date_col_name + "_" + i for i in self.added_cols_suffix])
        return self

    def transform(self,df,*_):
        copy = df.copy()
        for date_col_name in self.date_cols_names:
            copy[date_col_name] = pd.to_datetime(copy[date_col_name],format="%Y/%m/%d")
            temp = pd.DataFrame(index=copy.index)
            temp["Days_In_Month"] = copy[date_col_name].dt.days_in_month
            temp["last_day_minus_current"] = temp["Days_In_Month"] - copy[date_col_name].dt.day
            temp["current_minus_first_day"] = copy[date_col_name].dt.day - 1
            temp["First_Last_Few_Days_Of_Month"] = temp[["last_day_minus_current","current_minus_first_day"]].min(axis=1)
            copy[date_col_name+"_"+"Close_To_Month_Start_End"] = np.where(temp["First_Last_Few_Days_Of_Month"]<=5,1,0)
            

            copy[date_col_name+"_"+"Day"] = copy[date_col_name].dt.day
            copy[date_col_name+"_"+"Month"] = copy[date_col_name].dt.month
            copy[date_col_name+"_"+"Year"] = copy[date_col_name].dt.year
            copy[date_col_name+"_"+"Day_Name"] = copy[date_col_name].dt.day_name()
            copy[date_col_name+"_"+"Day_Num"] = copy[date_col_name].dt.weekday
            

            copy[date_col_name+"_"+"is_weekend"] = copy[date_col_name+"_"+"Day_Name"].isin(["Sunday","Saturday"]).map({True:1,False:0})
            
            copy[date_col_name+"_"+"is_holiday"] = copy[date_col_name].map(self.is_holiday)
            copy[date_col_name+"_"+"Quater"] = copy[date_col_name].map(self.get_quater)
   
        if self.return_whole_df:
            return copy[self.cols+self.added_cols]

        if self.drop_original_col:
            return copy[self.added_cols]

        return copy[self.date_cols_names + self.added_cols]

    def is_holiday(self,date):
        return date in holidays.India(years=date.year)
    
    def get_quater(self,date):
        if date.month < 4:
            return 1
        elif date.month < 7:
            return 2
        elif date.month < 10:
            return 3
        return 4
    
    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols

        if self.drop_original_col:
            return self.added_cols

        return self.date_cols_names + self.added_cols
    

class DateDiff(TransformerMixin,BaseEstimator):
    def __init__(self,date_col):
        self.cols = []
        self.date_col = date_col
        self.date_diff_col = self.date_col + "DateDiff"

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        self.cols.append(self.date_diff_col)
        
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        self.first_date = df[self.date_col].sort_values()[0].date()
        
        return self

    def transform(self,df,*_):
        df[self.date_diff_col] = df[self.date_col].map(lambda x: (x.date()-self.first_date).days)
        return df

    def get_feature_names(self):
        return self.cols

    
class NullHandler(TransformerMixin,BaseEstimator):
    """
        Adds a feature IsNull for all numeric columns
        Imputes Numeric Null values using median strategy of SimpleImputer
        Categorical Null values are replaced by "Null" representing a null category   
        Adds a feature Null_Pct which is the % of nulls in the given row
        Does not include columns ending with IsOutlier when calculating Null %
        Set exclude_cols_for_null_pct for excluding other columns if any
    """
    
    df = None
    cols = None
    null_cols = []
    null_num_cols = []
    null_cat_cols = []
    numeric_imputer = None
    
    def __init__(self,exclude_cols_for_null_pct=[]):
        """
         exclude_cols_for_null_pct : Pass column names which are to be excluded when calculating Null %.
        """
        self.exclude_cols_for_null_pct = exclude_cols_for_null_pct
    
    
    def fit(self,df,y=None):
        self.cols = list(df.columns)
        
        self.numeric_imputer = SimpleImputer(strategy="median")
        
        self._infer_dtypes(df)
        
        self.numeric_imputer.fit(df.loc[:,self.null_num_cols])
        
        return self

    def transform(self,df,*_):
        self.df = df.copy()
        
        self._null_pct()
        self._numeric_is_null()
        self._cat_null_cat()
        self._numeric_impute_null()
            
        return self.df
    
    def _numeric_impute_null(self):
        self.df[self.null_num_cols] = self.numeric_imputer.transform(self.df.loc[:,self.null_num_cols]) 
    
    def _numeric_is_null(self):
        for col in self.null_num_cols:
            self.df[col+"IsNull"] = self.df[col].isnull()
            
                
    def _cat_null_cat(self):
        for col in self.null_cat_cols:
            self.df.loc[self.df[self.df[col].isnull()].index.to_list(),col] = "Null"
    
    def _null_pct(self):
        cols_to_include = [i for i in self.cols if not i.endswith("IsOutlier")]
        cols_to_include = [i for i in cols_to_include if i not in self.exclude_cols_for_null_pct]
        
        self.df["Null_Pct"] = np.round(self.df[cols_to_include].isnull().mean(axis=1) * 100,2)
        
        
    def _infer_dtypes(self,df):
        self.null_cols = pd.DataFrame(df.isnull().sum(),columns=["Nulls"]).query("Nulls>0").index.to_list()
        for col in self.null_cols:
            dtype = df[col].dtype
            if dtype in ["int","float"]:
                self.null_num_cols.append(col)
            else:
                self.null_cat_cols.append(col)
            
    def get_feature_names(self):
        return list(self.cols) + ["Null_Pct"] +[i + "IsNull" for i in self.null_num_cols]
    
    
class OutlierHandler(TransformerMixin,BaseEstimator):
    """
        Adds a feature IsOutlier for all numeric columns
        Outlier is identified either by using standard deviation or
        IQR
    """
    df = None
    cols = None
    cols_gaussian_info = None # {col: True / False}
    boundaries = {} # {col: [lower,upper]}
    
    def __init__(self,cols_gaussian_info={},gaussian_threshold=2,iqr_threshold=1.5):
        """
            cols_gaussian_info : Pass column names and boolean value indicating whether the column follows a gaussian distribution or not.
            Syntax: cols_guassian_info = {'col_A' : True, 'col_B' : False}
            Default value for columns not passed is True (follows gaussina distribution)
        
            Gaussian Threshold: Mean +- Std * threshold
            IQR Threshold: 75th percentile + IQR * threshold
            
            Common Values for Gaussian Threshold: 2, 3
            Common Values for IQR Threshold: 1.5, 3
        """
        
        self.cols_gaussian_info = cols_gaussian_info
        self.gaussian_threshold = gaussian_threshold
        self.iqr_threshold = iqr_threshold
    
    
    def fit(self,df,y=None):
        self.cols = list(df.columns)  
        
        numeric = df.select_dtypes(include=np.number)
        self.numeric_cols = numeric.columns
        
        for col in self.numeric_cols:
            info = self.cols_gaussian_info.get(col,True)
            if info:
                # Followes Gaussian Distribution
                bounds = self._get_gaussian_boundaries(df[col])
            else:
                bounds = self._get_iqr_boundaries(df[col])
            self.boundaries[col] = bounds
        
        return self

    def transform(self,df,*_):
        self.df = df.copy()
        
        for col in self.numeric_cols:
            bounds = self.boundaries[col]
            self.df[col+"IsOutlier"] = self.df[col].map(lambda x: x < bounds[0] or x > bounds[1])
            
        return self.df
    
    def _get_gaussian_boundaries(self,col : pd.Series):
        mean = col.mean()
        three_std = col.std()*self.gaussian_threshold
        lower = mean-three_std
        upper = mean+three_std
        return [lower,upper]
    
    def _get_iqr_boundaries(self,col : pd.Series):
        IQR = iqr(col)
        lower = col.quantile(0.25) - IQR * self.iqr_threshold
        upper = col.quantile(0.75) + IQR * self.iqr_threshold
        return [lower,upper]
            
    def get_feature_names(self):
        return list(self.cols) + [i + "IsOutlier" for i in self.numeric_cols]
    
class PassThrough(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.cols = []

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        return self

    def transform(self,df,*_):
        return df

    def get_feature_names(self):
        return self.cols

class Dropper(TransformerMixin,BaseEstimator):
    def __init__(self,cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self,df,y=None):
        return self

    def transform(self,df,*_):
        return df.drop(self.cols_to_drop,axis=1)

    def get_feature_names(self):
        return self.cols
    
class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self._feature_names = feature_names 

    def fit(self, X, y = None):
        return self 

    def transform(self, X, y = None):
        return X[self._feature_names]