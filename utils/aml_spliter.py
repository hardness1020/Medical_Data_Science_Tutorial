import numpy as np
from sklearn.model_selection import train_test_split
from .selected_features import Features


class AMLSpliter(object):
    def __init__(self, df, reference_column, feature_column, label_column, feature_name=None):
        '''
        Drop the duplicates and missing values, then split the data into train and test set

        Needed parameters
        :param Dataframe df: the dataframe containing whole data.
        :param [] reference_column: the column that is used to identify the data.
        :param [] feature_column: the columns that are used as features.
        :param [] label_column: the column that is used as label.
        '''
        self.df = df[['cohort'] + reference_column + feature_column + label_column].copy()

        # drop samples with missing value
        self.df = self.df.dropna(subset=label_column + feature_column)
        
        # drop samples with duplicates
        self.df = self.df.drop_duplicates(subset=reference_column + feature_column)
        
        self.feature_column = feature_column
        self.label_column = label_column

    
    def train_test_split_and_normalization_by_cohort(self, cohorts, normalize=True, test_size=0.2, random_state=42):
        '''
        Split the data into train and test set by cohort
        '''
        X_trains, X_tests, y_trains, y_tests, train_indices, test_indices = [], [], [], [], [], []
        features = Features()
        
        for cohort in cohorts:
            df = self.df[self.df['cohort'] == cohort]

            # use index to retain the index of data in different cohorts
            df_index = df.index.values
            train_index, test_index = train_test_split(df_index, test_size=test_size, random_state=random_state)
            X_train, X_test, y_train, y_test = df.loc[train_index, self.feature_column], \
                                               df.loc[test_index, self.feature_column], \
                                               df.loc[train_index, self.label_column].astype('int'), \
                                               df.loc[test_index, self.label_column].astype('int')
            
            # preprocessing
            if normalize:
                features.preprocessor.fit(X_train)
                X_train[features.age_data + features.blood_data] = features.preprocessor.transform(X_train)
                X_test[features.age_data + features.blood_data] = features.preprocessor.transform(X_test)

            X_trains.append(X_train.values)
            X_tests.append(X_test.values)
            y_trains.append(y_train.values.ravel())
            y_tests.append(y_test.values.ravel())
            train_indices.append(train_index)
            test_indices.append(test_index)

        # Concatenate the data
        X_train = np.concatenate(X_trains, axis=0)
        X_test = np.concatenate(X_tests, axis=0)
        y_train = np.concatenate(y_trains, axis=0)
        y_test = np.concatenate(y_tests, axis=0)
        train_index = np.concatenate(train_indices, axis=0)
        test_index = np.concatenate(test_indices, axis=0)

        if normalize:
            return X_train, X_test, y_train, y_test, train_index, test_index, features.preprocessor
        else:
            return X_train, X_test, y_train, y_test, train_index, test_index, None
    