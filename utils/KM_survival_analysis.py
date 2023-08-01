import io
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


class KMSurvivalCurve:
    def __init__(self, df, time_column, event_column, label_column, title='', label_columns_name=['adverse (0)', 'intermediate (1)', 'favorable (2)']):
        '''
        Draw the Kaplan-Meier survival curve

        Needed parameters:
            :param Dataframe df: the dataframe.
            :param [] time_column: the column that stores survival or observation time.
            :param [] event_column: the column that stores if the sample trigger event (dead or others).
            :param [] label_column: the column that stores the label(group) of the sample. 0:adverse 1:intermediate 2:favorable.
            :param str title: the title of the plot.
            :param str label_columns_name: the name of each label.
        '''
        self.df = df.copy()
        self.time_column = time_column
        self.event_column = event_column
        self.label_column = label_column
        self.title = title
        self.label_columns_name = label_columns_name

    def return_survival_curve_bytes(self, plot=False):
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(6, 4))
        plt.grid(axis='y')
        
        pvalues = []
        for i in sorted(self.df[self.label_column].unique()):
            if self.label_columns_name is None:
                kmf.fit(self.df.loc[self.df[self.label_column] == i, self.time_column], 
                        self.df.loc[self.df[self.label_column] == i, self.event_column], 
                        label=i)
            else:
                kmf.fit(self.df.loc[self.df[self.label_column] == i, self.time_column], 
                        self.df.loc[self.df[self.label_column] == i, self.event_column], 
                        label=self.label_columns_name[int(i)])
            
            if i == 0:
                kmf.plot(color='#CE0000')
            elif i == 1:
                kmf.plot(color='#D9B300')
            elif i == 2:
                kmf.plot(color='#00BB00')
            
            # Calculate p-values
            for j in sorted(self.df[self.label_column].unique()):
                if (i==0 and j==1) or (i==1 and j==2):
                    pvalue = logrank_test(self.df.loc[self.df[self.label_column] == i, self.time_column],
                                          self.df.loc[self.df[self.label_column] == j, self.time_column],
                                          self.df.loc[self.df[self.label_column] == i, self.event_column],
                                          self.df.loc[self.df[self.label_column] == j, self.event_column]).p_value
                    pvalues.append((int(i), int(j), pvalue))
                    
        # Add p-values to the plot
        for idx, (label1, label2, pvalue) in enumerate(pvalues):
            plt.annotate(f'p-value ({label1}, {label2}): {pvalue:.1E}', xy=(0.20, 0.90 - idx * 0.05), xycoords='axes fraction')
                
        plt.title(self.title + ' Overall Survival Curve')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.ylim(-0.05, 1.05)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        img_bytes = buf.read()
        
        if plot:
            plt.show()
        return img_bytes

    def get_pvalue_by_label(self, label1, label2):
        group1 = self.df.loc[self.df[self.label_column] == label1, self.time_column].copy()
        group2 = self.df.loc[self.df[self.label_column] == label2, self.time_column].copy()
        event1 = self.df.loc[self.df[self.label_column] == label1, self.event_column].copy()
        event2 = self.df.loc[self.df[self.label_column] == label2, self.event_column].copy()
        pvalue = logrank_test(group1, group2, event1, event2).p_value
        return pvalue


