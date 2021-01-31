import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator
from scipy.stats import iqr, shapiro
import holidays

class Common:
    def get_cols_needed(self,cols,include=None,exclude=None):
        if include == None and exclude == None:
            return cols
        
        if exclude == None:
            return include
        
        if include == None:
            return [col for col in cols if col not in exclude]
        
        raise Exception("Either specify include or exclude or None. Both cannot be specified")
        
    def return_transformed_df(self,df,return_whole_df,drop_original_col,cols,added_cols,processed_cols):
        cols = set(cols)
        cols_to_remove = set(processed_cols)
        
        if return_whole_df:
            
            if drop_original_col:
                cols_to_include = list(cols.difference(processed_cols))
                return df[cols_to_include + added_cols]
            
            return df[list(cols) + added_cols]
        
        if drop_original_col:
            return df[added_cols]
        
        return df[processed_cols + added_cols]

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
        self.cols = list(df.columns)
        return self

    def transform(self,df,*_):
        return df.drop(self.cols_to_drop,axis=1)

    def get_feature_names(self):
        return list(set(self.cols).difference(set(self.cols_to_drop)))
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    """
        Used to select only certain columns
    """
    def __init__(self, feature_names):
        """
            feature_names : column names to be selected
        """
        
        self._feature_names = feature_names 

    def fit(self, X, y = None):
        return self 

    def transform(self, X, y = None):
        return X[self._feature_names]
    
    def get_feature_names(self):
        return self._feature_names
    

class ColumnNameApplyer(TransformerMixin,BaseEstimator):
    """
        Transforms the passed argument into a dataframe and applies the column names
    """
    def __init__(self,column_names):
        """
            column_names : Column Names to be applied
        """
        
        self.column_names=column_names
        

    def fit(self,df,y=None):
        return self

    def transform(self,X,*_):
        copy = pd.DataFrame(X,columns=self.column_names)
        return copy

    def get_feature_names(self):
        return self.column_names

class DTypeTransformer(TransformerMixin,BaseEstimator):
    """
        Converts columns as per the given dtypes
    """
    def __init__(self, mapping={}):
        """
        Specify the dtype of column(s) you want to convert to\n
        possible values are : int64, float64, bool, category, dt_fmt, str
        dt_fmt : convert to datetime where fmt represents the format to parse
        """
        self.mapping = mapping

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        return self

    def transform(self,df,*_):
        copy = df.copy()
        
        for col,dtype in self.mapping.items():
            if dtype == "int64":
                copy[col] = copy[col].astype(np.int64)
            
            elif dtype == "float64":
                copy[col] = copy[col].astype(np.float64)
                
            elif dtype == "bool":
                copy[col] = copy[col].astype(bool)
            
            elif dtype == "category":
                copy[col] = pd.Categorical(copy[col])
                
            elif dtype == "str":
                copy[col] = copy[col].astype(str)
            
            elif dtype.startswith("dt_"):
                fmt = dtype[3:]
                copy[col] = pd.to_datetime(copy[col],format=fmt)
            
            else:
                raise Exception("Unsupported dtype specified")
                
        return copy
    

    def get_feature_names(self):
        return self.cols
    
    @staticmethod
    def get_mapping_from_map_of_lists(ints=[],floats=[],bools=[],cats=[],strs=[]):
        """
            Returns a mapping of dtypes generated using the column names passed in the dtypes lists
        """
        mapping = {}
        
        for cols,dtype in zip([ints,floats,bools,cats,strs],["int64","float64","bool","category","str"]):
            mapping.update({col:dtype for col in cols})
        
        return mapping

class NullPct(TransformerMixin,BaseEstimator):
    """
        Adds a feature null_pct which is the % of nulls in the given row
        Does not include columns ending with _is_outlier when calculating Null %
        Set exclude_cols_for_null_pct for excluding other columns if any
    """
    
    df = None
    cols = None
    
    def __init__(self,exclude_cols_for_null_pct=[]):
        """
         exclude_cols_for_null_pct : Pass column names which are to be excluded when calculating Null %.
        """
        self.exclude_cols_for_null_pct = exclude_cols_for_null_pct
    
    
    def fit(self,df,y=None):
        self.cols = list(df.columns)
        
        return self

    def transform(self,df,*_):
        self.df = df.copy()
        
        self._null_pct()
            
        return self.df
    
    
    def _null_pct(self):
        cols_to_include = [i for i in self.cols if not i.endswith("_is_outlier")]
        cols_to_include = [i for i in cols_to_include if i not in self.exclude_cols_for_null_pct]
        
        self.df["null_pct"] = np.round(self.df[cols_to_include].isnull().mean(axis=1) * 100,2)
        
        
    def get_feature_names(self):
        return list(self.cols) + ["null_pct"] 
    

class IsNull(TransformerMixin,BaseEstimator,Common):
    """
        Adds a missingness indicator for all records in columns that have atleast one null value during fitting
        Column null_pct and all columns ending with _is_outlier are excluded
    """
    def __init__(self,return_whole_df= True,drop_original_col = True,include=None,exclude=None):
        """
        include : features names to be generated
        exclude : feature names to be excluded while generating all features
        return_whole_df : whether to to return the whole dataframe
        drop_original_col : whether to drop the columns using which processing is done
        """
        self.return_whole_df = return_whole_df
        self.drop_original_col = drop_original_col
        self.include = include
        self.exclude = exclude
        
    def fit(self,df,y=None):
        self.cols = list(df.columns)
        self.included_cols = self.get_cols_needed(self.cols,self.include,self.exclude)
        self.included_cols = [i for i in self.included_cols if not i.endswith("_is_outlier") and i!="null_pct"]
        self.added_cols = [i + "_is_null" for i in self.included_cols]
        
        return self
    
    def transform(self,df,*_):
        copy = df.copy()
        copy[self.added_cols] = df[self.included_cols].isnull()
        
        return self.return_transformed_df(copy,self.return_whole_df,self.drop_original_col,self.cols,self.added_cols,self.included_cols)
        
    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols
        return self.added_cols

class OutlierDetector(TransformerMixin,BaseEstimator, Common):
    """
        Adds a feature IsOutlier for all numeric columns
        Outlier is identified either by using standard deviation or
        IQR
    """
    df = None
    cols = None
    cols_gaussian_info = None # {col: True / False}
    boundaries = {} # {col: [lower,upper]}
    
    def __init__(self,auto_infer_whether_guassian_dist=True, cols_gaussian_info={},gaussian_threshold=2,iqr_threshold=1.5,include=None,exclude=None):
        """
            include : features names to be generated
            exclude : feature names to be excluded while generating all features
            auto_infer_whether_guassian_dist : Automaticaly infer whether the feature follows guassian distribution for each feature (cols_gaussian_info should be empty)
            cols_gaussian_info : Pass column names and boolean value indicating whether the column follows a gaussian distribution or not (auto_infer_whether_guassian_dist should be False if this is not empty)
            Syntax: cols_guassian_info = {'col_A' : True, 'col_B' : False}
            Default value for columns not passed is True (follows gaussian distribution)
        
            Gaussian Threshold: Mean +- Std * threshold
            IQR Threshold: 75th percentile + IQR * threshold
            
            Common Values for Gaussian Threshold: 2, 3
            Common Values for IQR Threshold: 1.5, 3
        """
        self.auto_infer_whether_guassian_dist = auto_infer_whether_guassian_dist
        self.cols_gaussian_info = cols_gaussian_info
        
        if self.auto_infer_whether_guassian_dist and self.cols_gaussian_info != {}:
            raise Exception("Either auto_infer_whether_guassian_dist should be true and cols_gaussian_info should be empty\n or auto_infer_whether_guassian_dist should be false and cols_gaussian_info should be not empty")
        
        if (not self.auto_infer_whether_guassian_dist) and self.cols_gaussian_info == {}:
            raise Exception("Either auto_infer_whether_guassian_dist should be true and cols_gaussian_info should be empty\n or auto_infer_whether_guassian_dist should be false and cols_gaussian_info should be not empty")
        
        self.gaussian_threshold = gaussian_threshold
        self.iqr_threshold = iqr_threshold
        
        self.include = include
        self.exclude = exclude
    
    
    def fit(self,df,y=None):
        self.cols = list(df.columns)  
        
        numeric = df.select_dtypes(include=[int,float],exclude=[bool])
        self.numeric_cols = numeric.columns
        self.included_cols = self.get_cols_needed(self.numeric_cols,self.include,self.exclude)
        
        for col in self.included_cols:
            if self.auto_infer_whether_guassian_dist:
                info = self._is_gaussian(numeric[col])
            else:
                info = self.cols_gaussian_info.get(col,True)
            if info:
                # Follows Gaussian Distribution
                bounds = self._get_gaussian_boundaries(df[col])
            else:
                bounds = self._get_iqr_boundaries(df[col])
            self.boundaries[col] = bounds
        
        return self

    def transform(self,df,*_):
        self.df = df.copy()
        
        for col in self.included_cols:
            bounds = self.boundaries[col]
            self.df[col+"_is_outlier"] = self.df[col].map(lambda x: x < bounds[0] or x > bounds[1])
            
        return self.df
    
    def _is_gaussian(self,col):
        stat, p = shapiro(col)
        if p >= 0.05:
            return True
        return False
    
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
        return list(self.cols) + [i + "_is_outlier" for i in self.included_cols]
    
class OutlierImputer(TransformerMixin,BaseEstimator):
    """
        Imputes outliers by the value specified for each column
        For each column, a feature named featureName_is_outlier must be present in the dataframe passed
    """
    def __init__(self,impute_values = {}):
        """
            impute_values = {column_name : impute_value}
        """
        self.impute_values=impute_values
        self.column_names = []

    def fit(self,df,y=None):
        self.column_names = list(df.columns)
        return self

    def transform(self,df,*_):
        copy = df.copy()
        
        for col,impute_value in self.impute_values.items():
            copy.loc[df[col+"_is_outlier"]==True,col] = impute_value
        
        return copy

    def get_feature_names(self):
        return self.column_names

class DateDiff(TransformerMixin,BaseEstimator,Common):
    """
    Considering the earliest date as 1 transforms all dates and adds a feature (kind of reference pointer)
    """
    
    def __init__(self,date_cols_names:list,return_whole_df = True,drop_original_col = True,date_format = "%Y/%m/%d"):
        """
            date_cols_names : column names having date
            date_format : the date format (strpformat)
            return_whole_df : whether to to return the whole dataframe
            drop_original_col : whether to drop the columns using which processing is done
        """
        self.cols = []
        self.date_cols_names = date_cols_names
        
        self.return_whole_df = return_whole_df
        self.drop_original_col = drop_original_col
        self.date_format = date_format
        
        self.added_cols = []
        self.first_dates = {}

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        
        copy = df.copy()
        for date_col_name in self.date_cols_names:
            if self.drop_original_col:
                self.cols.remove(date_col_name)
            self.added_cols.append(date_col_name + "_date_diff")
            
            copy[date_col_name] = pd.to_datetime(copy[date_col_name],format=self.date_format)
            first_date = copy[date_col_name].sort_values().iloc[0].date()
            self.first_dates[date_col_name] = first_date
        
        
        return self

    def transform(self,df,*_):
        copy = df.copy()
        for date_col_name in self.date_cols_names:
            copy[date_col_name] = pd.to_datetime(copy[date_col_name],format=self.date_format)
            copy[date_col_name+"_date_diff"] = copy[date_col_name].map(lambda x: (x.date()-self.first_dates.get(date_col_name)).days)
        
        return self.return_transformed_df(copy,self.return_whole_df,self.drop_original_col,self.cols,self.added_cols,self.date_cols_names)

    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols

        if self.drop_original_col:
            return self.added_cols

        return self.date_cols_names + self.added_cols

class DateHandler(TransformerMixin,BaseEstimator,Common):
    """ 
        Splits the date and
        Creates features such as
        day : 1 - 131
        month : 1 - 12
        year : the year
        day_name : Monday to Sunday
        day_num : 0 to 6
        quater : Quater of the year decided by month
        is_weekend : Whether there is a weekend for given date
        is_holiday : Whether there is a holiday for given date for country specified (holidays (https://pypi.org/project/holidays/) library is used)
        close_to_start_month_end : Boolean indicating whether the date is close to month's start or end (True means close)
    """

    def __init__(self,date_cols_names:list,return_whole_df = True,drop_original_col = True,date_format = "%Y/%m/%d",close_to_start_month_end_param = 5,country="India",include=None,exclude=None):
        """
            date_cols_names : column names having date
            date_format : the date format (strpformat)
            include : features names to be generated
            exclude : feature names to be excluded while generating all features
            return_whole_df : whether to to return the whole dataframe
            drop_original_col : whether to drop the columns using which processing is done
            close_to_start_month_end_param: No. of days specifiying how close is a given date close to start, end of a month
            country : the country name for which holidays are considered
        """
        self.return_whole_df = return_whole_df
        self.cols = []
        self.date_cols_names = date_cols_names
        self.added_cols_suffix = ["day","month","year","day_name","day_num","is_weekend","close_to_month_start_end","is_holiday","quater"]
        self.added_cols = []
        self.drop_original_col = drop_original_col
        self.date_format = date_format
        self.close_to_start_month_end_param = close_to_start_month_end_param
        self.include = include
        self.exclude = exclude
        self.country = country


    def fit(self,df,y=None):
        self.cols = list(df.columns)
        self.included_cols = self.get_cols_needed(self.added_cols_suffix,self.include,self.exclude)
        for date_col_name in self.date_cols_names:
            if self.drop_original_col:
                self.cols.remove(date_col_name)
            self.added_cols.extend([date_col_name + "_" + i for i in self.added_cols_suffix if i in self.included_cols])
        return self

    def transform(self,df,*_):
        copy = df.copy()
        for date_col_name in self.date_cols_names:
            copy[date_col_name] = pd.to_datetime(copy[date_col_name],format=self.date_format)
            
            if "close_to_month_start_end" in self.included_cols:
                temp = pd.DataFrame(index=copy.index)
                temp["Days_In_Month"] = copy[date_col_name].dt.days_in_month
                temp["last_day_minus_current"] = temp["Days_In_Month"] - copy[date_col_name].dt.day
                temp["current_minus_first_day"] = copy[date_col_name].dt.day - 1
                temp["First_Last_Few_Days_Of_Month"] = temp[["last_day_minus_current","current_minus_first_day"]].min(axis=1)
                copy[date_col_name+"_close_to_month_start_end"] = np.where(temp["First_Last_Few_Days_Of_Month"]<=self.close_to_start_month_end_param,True,False)
            
            if "day" in self.included_cols:
                copy[date_col_name+"_day"] = copy[date_col_name].dt.day
                
            if "month" in self.included_cols:    
                copy[date_col_name+"_month"] = copy[date_col_name].dt.month
                
            if "year" in self.included_cols:
                copy[date_col_name+"_year"] = copy[date_col_name].dt.year
                
            if "day_name" in self.included_cols:
                copy[date_col_name+"_day_name"] = copy[date_col_name].dt.day_name()
                
            if "day_num" in self.included_cols:
                copy[date_col_name+"_day_num"] = copy[date_col_name].dt.weekday
                
            if "is_weekend" in self.included_cols:
                copy[date_col_name+"_is_weekend"] = copy[date_col_name].dt.day_name().isin(["Sunday","Saturday"])
                
            if "is_holiday" in self.included_cols:
                copy[date_col_name+"_is_holiday"] = copy[date_col_name].map(self.is_holiday)
                
            if "quater" in self.included_cols:
                copy[date_col_name+"_quater"] = copy[date_col_name].map(self.get_quater)
   
        return self.return_transformed_df(copy,self.return_whole_df,self.drop_original_col,self.cols,self.added_cols,self.date_cols_names)

    def is_holiday(self,date):
        return date in holidays.CountryHoliday(country=self.country,years=date.year)
    
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

class TimeHandler(TransformerMixin,BaseEstimator,Common):
    """ Splits the time and
        Creates features such as
        hour : 0 - 23
        minute : 0-59
        period_name : Morning, Afternoon, Evening, Night
        period_num : 1,2,3,4
        military_time : 0000 to 2359 
    """
    def __init__(self,time_cols_names:list,return_whole_df= True,drop_original_col = True,time_format="%H:%M",include=None,exclude=None):
        """
            time_cols_names : column names having time
            time_format : the time format (strpformat)
            include : features names to be generated
            exclude : feature names to be excluded while generating all features
            return_whole_df : whether to to return the whole dataframe
            drop_original_col : whether to drop the columns using which processing is done
        """
        
        self.return_whole_df = return_whole_df
        self.cols = []
        self.time_cols_names = time_cols_names
        self.added_cols_suffix = ["hour","minute","military_time","period_name","period_num"]
        self.added_cols = []
        self.drop_original_col = drop_original_col
        self.time_format = time_format
        self.include = include
        self.exclude = exclude

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        self.included_cols = self.get_cols_needed(self.added_cols_suffix,self.include,self.exclude)
        for time_col_name in self.time_cols_names:
            if self.drop_original_col:
                self.cols.remove(time_col_name)
            self.added_cols.extend([time_col_name + "_" + i for i in self.added_cols_suffix if i in self.included_cols])
        
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
            copy[time_col_name] = pd.to_datetime(copy[time_col_name],format=self.time_format)

            if "period_name" in self.included_cols or "period_num" in self.included_cols:
            
                copy[time_col_name+"_hour"] = copy[time_col_name].dt.hour
                copy[time_col_name+"_minute"] = copy[time_col_name].dt.minute
                
                # Military Time
                copy[time_col_name+"_military_time"] = (copy[time_col_name+"_hour"].astype(str) + copy[time_col_name+"_minute"].astype(str)).astype(int)
                
                if "period_name" in self.included_cols:
                    # Time Period
                    copy[time_col_name+"_period_name"] = copy[time_col_name+"_military_time"].apply(self.get_time_period_name)
                
                if "period_num" in self.included_cols:
                    copy[time_col_name+"_period_num"] = copy[time_col_name+"_military_time"].apply(self.get_time_period_num)
                
                if "military_time" not in self.included_cols:
                    copy.drop(time_col_name+"_military_time",axis=1,inplace=True)
            
            else:
                if "hour" in self.included_cols:
                    copy[time_col_name+"_hour"] = copy[time_col_name].dt.hour
                    
                if "minute" in self.included_cols:
                    copy[time_col_name+"_minute"] = copy[time_col_name].dt.minute
                
                if "military_time" in self.included_cols:
                    copy[time_col_name+"_military_time"] = (copy[time_col_name+"_hour"].astype(str) + copy[time_col_name+"_minute"].astype(str)).astype(int)
            
        
        return self.return_transformed_df(copy,self.return_whole_df,self.drop_original_col,self.cols,self.added_cols,self.time_cols_names)

    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols
        return self.added_cols

