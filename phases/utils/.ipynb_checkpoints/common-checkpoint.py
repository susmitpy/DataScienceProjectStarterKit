from IPython.display import display_html
import re
import pandas as pd

def display_side_by_side(dfs,mssg=None):
    html_str=''
    for index,df in enumerate(dfs):
        if mssg:
            html_str+=df.to_html().replace("<thead",f"<caption>{mssg[index]}</caption>\n<thead")
        else:
            html_str+=df.to_html()
        
        
    display_html(html_str.replace('table','table style="display:inline;padding:12px;margin:6px;padding-top:10px;"'),raw=True)
    
def show_category_summary(df: pd.DataFrame, category_cols: list,target_col: str,target_col_dtype: str):
    """
        df : Dataframe
        category_cols : list of category columns
        target_col_dtype : num or cat
        target_col : name of target column
    """
    assert target_col_dtype in ["num","cat"], "target_col_dtype must be either num or cat"
    
    vcs = []
    mssgs = []
    for col in category_cols:
        if target_col_dtype == "cat":
            agg_func = "mean"
            col_prefix = "% "
        if target_col_dtype == "num":
            agg_func = "median"
            col_prefix = "Median "

        grouped = pd.DataFrame(df.groupby(col)[target_col].agg(agg_func))
        grouped.columns = [col_prefix + target_col]

        vc = pd.DataFrame(df[col].value_counts(dropna=False,normalize=True))
        vc.columns = ["Count %"]
        ans = pd.concat([vc.sort_index(),grouped.sort_index()],axis=1)
        ans.index.name = col
        ans.sort_values("Count %",ascending=False)
        vcs.append(ans.head(5))
        mssgs.append(f"Unique Categories : {len(vc)}")

    display_side_by_side(vcs,mssgs)
    
def display_dtypes(df,num_rows_per_column=10):
    dfs = []
    l = len(df.columns)
    dfdtypes = pd.DataFrame(df.dtypes)

    print("DTypes")
    for i in range(l//num_rows_per_column+1):
        dfs.append(dfdtypes.iloc[i*num_rows_per_column : (i+1)*num_rows_per_column])

    display_side_by_side(dfs,mssg=None)

def get_original_column_names(cols):
    """
    Given column names returned by ColumnTransformer returns original column names
    """
    pattern = r"([a-zA-Z]__)(.+)"
    return [re.search(pattern,i).groups()[-1] for i in cols]