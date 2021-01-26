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