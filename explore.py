import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# remove warnings
import warnings
warnings.filterwarnings("ignore")


def plot_categorical_and_continuous_vars(df, cat_vars, cont_vars):
    for cat in cat_vars:
        for cont in cont_vars:
            # creates a boxplot
            sns.boxplot(x=cat, y=cont, data=df)
            plt.title('Distribution of ' + cont)
            plt.show()
            
            # creates a swarmplot
            sns.swarmplot(x=cat, y=cont, data=df)
            plt.show()
    
            # creates a stripplot
            sns.stripplot(x=cat, y=cont, data=df)
            plt.show()
            plt.tight_layout()

def plot_variable_pairs(df):
    new_df = df.drop(columns=['customer_id'])
    cols = new_df.columns.to_list()
    # show pairwise relationships
    sns.pairplot(new_df[cols], corner=True, kind="reg", plot_kws={'line_kws':{'color':'red'}})
    plt.show()