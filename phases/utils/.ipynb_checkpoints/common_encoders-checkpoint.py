from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class CommonOrdinalEncoder(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.mapping = {}
        self.cols=[]

    def fit(self,df,y=None):
        self.cols=df.columns
        i = 0
        for col in self.cols:
            self.mapping[col] = {}
            for unique_value in list(df[col].unique()):
                self.mapping[col][unique_value] = i
                i += 1
            i = 0
        return self
    
    def apply_mapping(self,col_value,col_name):
        return int(self.mapping[col_name].get(col_value,-1))
    
    def transform(self,df,*_):
        copy = df.copy()
        for col in self.cols:
            copy[col] = df[col].apply(self.apply_mapping,args=[col])
            copy[col] = pd.Categorical(copy[col])

        return copy[self.cols]
    
    def get_feature_names(self):
        return self.cols


class CommonOHE(TransformerMixin,BaseEstimator):
    def __init__(self,cols):
        self.df_data = {}
        self.cols = {}
        
    def fit(self,df,y=None):
        self.df_data = {}
        self.cols = {}
        for col in self.enc_cols:
            for unique_value in list(df[col].unique())[:-1]:
                self.df_data[f"{col}_{unique_value}"] = []
                self.cols[f"{col}_{unique_value}"] = True
                
        return self
    
    def apply_mapping(self,col_value,col_name):
        col = f"{col_name}_{col_value}"
        for k in list(self.cols.keys()):
            if k.split("_")[0] != col_name:
                continue
            if k == col:
                self.df_data[k].append(1)
            else:
                self.df_data[k].append(0)
        return col_value
    
    def transform(self,df,*_):
        copy = df.copy()
        for col in self.enc_cols:
            df[col].apply(self.apply_mapping,args=[col])
            copy.drop(col,axis=1,inplace=True)
        if self.return_whole_df:
            return copy.join(pd.DataFrame(self.df_data,index=df.index))
        return pd.DataFrame(self.df_data,index=df.index)

        
