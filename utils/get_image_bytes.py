import io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .KM_survival_analysis import KMSurvivalCurve


# Define get confusion matrix bytes and 
def get_confusion_matrix_bytes(y_test, y_pred, title):
    cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, normalize='true'), 
                                     display_labels=['adverse', 'intermediate', 'favorable'])
    ax = plt.figure(figsize=(5,4)).subplots()
    ax.set(title=title)
    cmp.plot(ax=ax, cmap=plt.cm.OrRd)
    
    # store confusion matrix as bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    confusion_matrix_bytes = buf.read()
    buf.close()
    return confusion_matrix_bytes


# Define get survival curve bytes funciton
def get_survival_curve_bytes(y_pred, original_df, test_index, time_column, event_column, title='OS', cohort=None):
    '''
    Draw the Kaplan-Meier survival curve, 
    if cohort is not None, then only draw the curve for the cohort

    Needed parameters:
        :param dict y_pred              : the predicted label of the sample.
        :param Dataframe original_df    : the dataframe.
        :param [] test_index            : the index of y_pred data in original_df
        :param [] time_column           : the column that stores survival or observation time.
        :param [] event_column          : the column that stores if the sample trigger event (dead or others).
    Optional parameters:    
        :param str title                : the title of the plot.
        :param str cohort                : to get specific cohort of data in original_df
    '''

    if cohort == None:
        df_pred = pd.DataFrame(y_pred, columns=['label'])
        df_pred['os_coding'] = original_df.loc[test_index, event_column].values
        df_pred['os_survival'] = original_df.loc[test_index, time_column].values
    else:
        idx = original_df.loc[test_index, 'cohort'] == cohort
        
        df_pred = pd.DataFrame()
        df_pred['os_coding'] = original_df.loc[test_index, event_column].values[idx]
        df_pred['os_survival'] = original_df.loc[test_index, time_column].values[idx]

        idx = idx.reset_index(drop=True)
        df_pred['label'] = y_pred[idx]
        # title = cohort + ' ' + title

    OS_curve = KMSurvivalCurve(df=df_pred, title=title,
                               time_column='os_survival', event_column='os_coding', label_column='label',  
                               label_columns_name=['adverse (0)', 'intermediate (1)', 'favorable (2)'])
    survival_curve_bytes = OS_curve.return_survival_curve_bytes()
    return survival_curve_bytes  