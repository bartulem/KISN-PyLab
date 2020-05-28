# -*- coding: utf-8 -*-

"""

@author: bartulem

Perform linear regression on train/test split dataset.

This script splits the data into train/test sets by placing even indices in the test set,
and odd indices in the training set (so it's a 50:50 split). It performs a linear regression
on the training set and then predicts the test set values. It returns: (1) y_test, (2) y_test_predictions,
(3) slope and (4) intercept values (and possibly (5) extra data predictions if they are given as an input).
The function can take two keyword arguments as input: the xy_order (which column is X and which is Y),
and extra_data to be predicted by the model.

"""

import numpy as np
import pandas as pd
from sklearn import linear_model


class LinRegression:

    # initializer / instance attributes
    def __init__(self, input_data):
        self.input_data = input_data

    def split_train_test_and_regress(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        xy_order : list
            Determines what column is X and what column is Y data (e.g. [0, 1] would mean first column is X and the second is Y); defaults to [0, 1].
        extra_data : array
            Any additional data that requires to be predicted by the model; defaults to 0.
        ----------
        """

        xy_order = [kwargs['xy_order'] if 'xy_order' in kwargs.keys() and type(kwargs['xy_order']) == list and len(kwargs['xy_order']) == 2 else [0, 1]][0]
        extra_data = [kwargs['extra_data'] if 'extra_data' in kwargs.keys() else 0][0]

        # check if the input dataframe has nans, and if so - eliminate those rows
        if self.input_data.isnull().values.any():
            print('{} row(s) has/have NAN values and will be removed.'.format(self.input_data.isnull().any(axis=1).sum()))
            self.input_data = pd.DataFrame.dropna(self.input_data)

        split_df = {'x_train': [], 'x_test': [], 'y_train': [], 'y_test': []}

        # even indices are test, odd indices are train data
        for indx in range(self.input_data.shape[0]):
            if indx % 2 == 0:
                split_df['x_test'].append(self.input_data.iloc[indx, xy_order[0]])
                split_df['y_test'].append(self.input_data.iloc[indx, xy_order[1]])
            else:
                split_df['x_train'].append(self.input_data.iloc[indx, xy_order[0]])
                split_df['y_train'].append(self.input_data.iloc[indx, xy_order[1]])

        regress_data = {key: np.array(val).reshape((len(val), 1)) for key, val in split_df.items()}

        # train model
        lm = linear_model.LinearRegression()
        model = lm.fit(regress_data['x_train'], regress_data['y_train'])

        # predict test data
        return_dict = {'y_test': np.ravel(regress_data['y_test']), 'y_test_predictions': np.ravel(lm.predict(regress_data['x_test'])),
                       'slope': np.ravel(model.coef_)[0], 'intercept': np.ravel(model.intercept_)[0]}

        # predict other data if required
        if type(extra_data) != int:
            reshaped_extra_data = extra_data.reshape(extra_data.shape[0], 1)
            return_dict['extra_data_predictions'] = np.ravel(lm.predict(reshaped_extra_data))

        return return_dict
