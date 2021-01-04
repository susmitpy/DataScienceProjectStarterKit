from IPython.display import display_html

def display_side_by_side(mssg=None,*args):
    html_str=''
    for index,df in enumerate(args):
        if mssg:
            html_str+=df.to_html().replace("<thead",f"<caption>{mssg[index]}</caption>\n<thead")
        else:
            html_str+=df.to_html()
        
        
    display_html(html_str.replace('table','table style="display:inline;padding:12px;margin:6px;padding-top:10px;"'),raw=True)
