from IPython.display import display_html
import re
import pandas as pd

def subtract_two_lists(a:list=[],b:list=[]) -> list:
    """
        returns a list of elements in a but not in b
    """
    
    return list(set(a).difference(set(b)))

def get_original_cat_cols(cols:list=[]):
    """
        Returns columns which don't end with _is_null or _is_outlier
    """
    return [i for i in cols if not (i.endswith("_is_null") or i.endswith("_is_outlier"))]

def head(df,show=True):
    """
        Uses pandas dataframe head method, transposes the output
        displays using display() if show = True, else returns it
    """
    if show:
        display(df.head().transpose())
    else:
        return df.head().transpose()

def tail(df,show=True):
    """
        Uses pandas dataframe tail method, transposes the output
        displays using display() if show = True, else returns it
    """
    if show:
        display(df.tail().transpose())
    else:
        return df.tail().transpose()
    
def sample(df,show=True,n=None,frac=None,axis=None):
    """
        Uses pandas dataframe sample method, transposes the output
        Passes arguments n,frac,axis to the sample method
        displays using display() if show = True, else returns it
    """
    if show:
        display(df.sample(n=n,frac=frac,axis=axis).transpose())
    else:
        return df.sample(n=n,frac=frac,axis=axis).transpose()


def display_side_by_side(dfs,mssg=None):
    html_str=''
    for index,df in enumerate(dfs):
        if mssg:
            html_str+=df.to_html().replace("<thead",f"<caption>{mssg[index]}</caption>\n<thead")
        else:
            html_str+=df.to_html()
        
        
    display_html(html_str.replace('table','table style="display:inline;padding:12px;margin:6px;padding-top:10px;"'),raw=True)
    
def show_category_summary(df: pd.DataFrame, category_cols: list,target_col: str,target_col_dtype: str,normalize_over="columns"):
    """
        df : Dataframe
        category_cols : list of category columns
        target_col_dtype : num or cat
        target_col : name of target column
        normalize_over : for multi - class classification this argument is passed to crosstab() normalize method
    """
    assert target_col_dtype in ["num","cat"], "target_col_dtype must be either num or cat"
    
    vcs = []
    mssgs = []
    if target_col_dtype == "cat":
        nunique = df[target_col].nunique()
    for col in category_cols:
        if target_col_dtype == "cat":
            agg_func = "mean"
            col_prefix = "% "
            if nunique > 2:
                tab = pd.crosstab(df[col],df[target_col],normalize=normalize_over) * 100
                vcs.append(tab)
                mssgs.append("")
                continue
            
        if target_col_dtype == "num":
            agg_func = "median"
            col_prefix = "Median "

        grouped = pd.DataFrame(df.groupby(col)[target_col].agg(agg_func))*100
        grouped.columns = [col_prefix + target_col]

        vc = pd.DataFrame(df[col].value_counts(dropna=False,normalize=True)) * 100
        vc.columns = ["Count %"]
        ans = pd.concat([vc.sort_index(),grouped.sort_index()],axis=1)
        ans.index.name = col
        ans = ans.sort_values("Count %",ascending=False)
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
