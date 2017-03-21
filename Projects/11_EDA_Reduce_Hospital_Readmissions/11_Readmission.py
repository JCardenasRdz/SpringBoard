import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.plotly as py

def load_and_clean():
    column_names = ['Number of Discharges', 'Excess Readmission Ratio', 'Number of Readmissions', 'State']
    df = hospital_read_df = pd.read_csv('data/cms_hospital_readmissions.csv', skip_blank_lines = True)
    df = df[column_names]
    df.columns=['Discharges', 'Excess_Ratio', 'Readmissions', 'State']
    df = df.dropna()
    df.Discharges = df.Discharges.astype(float)
    df = df.iloc[np.isreal(df.Discharges),:]
    return df
    
def myscatter(df):
    plt.figure(figsize=(12,7))
    plt.subplot(1,3,1);
    plt.scatter(df.Discharges.apply(np.log10), df.Readmissions.apply(np.log10), c=df.Excess_Readmission_Ratio)
    plt.colorbar()
    plt.xlabel('Log 10 of Discharges')
    plt.ylabel('Log 10 of Readmissions')
    plt.title('Excess Readmission Ratio')

    plt.subplot(1,3,2);
    bool_ = df.Excess_Readmission_Ratio < 1.0
    plt.scatter(df.Discharges.apply(np.log10)[bool_], df.Readmissions.apply(np.log10)[bool_], c=df.Excess_Readmission_Ratio[bool_])
    plt.colorbar()
    plt.xlabel('Log 10 of Discharges')
    plt.title('Excess Readmission Ratio < 1.0')

    plt.subplot(1,3,3);
    bool_ = df.Excess_Readmission_Ratio > 1.0
    plt.scatter(df.Discharges.apply(np.log10)[bool_], df.Readmissions.apply(np.log10)[bool_], c=df.Excess_Readmission_Ratio[bool_])
    plt.colorbar()
    plt.xlabel('Log 10 of Discharges')
    plt.title('Excess Readmission Ratio > 1.0')

    plt.show()
    
def linear_fitting(df_, xdata, ydata):
    sns.regplot( x = xdata, y = ydata, data = df_);
    mod = sm.OLS( df_[ydata].values, df_[xdata].values )
    res = mod.fit()
    print(res.summary2())

def transform_data(df_,columns,func):
    for col in columns:
        df_[col] = df_[col].apply(func)
    return df_
    
def transformed_regression(dframe,func,label):
    dframe['Discharges'] = dframe['Discharges'].apply(func)
    linear_fitting( dframe  )
    plt.xlabel(label)
    
def usa_map_figure(df_,column, legend):
    for col in df_.columns:
        df_[col] = df_[col].astype(str)
    #scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
     #       [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

    data = [ dict(
        type='choropleth',
        #colorscale = scl,
        autocolorscale = True,
        locations = df_['State'],
        z = df_[column].astype(float),
        locationmode = 'USA-states',
       # text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = legend)
        ) ]

    layout = dict(
        title = ' ',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

    fig = dict( data=data, layout=layout )
    return fig
    #py.iplot( fig, filename='Excess-Readmission-Ratio' )
    
    
    