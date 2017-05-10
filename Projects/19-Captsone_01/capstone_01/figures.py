import seaborn as sns

def boxplot(ydata,xdata, hue_data, df):
    sns.boxplot(x= xdata, y=ydata, hue = hue_data, data=df, palette="Set3");
