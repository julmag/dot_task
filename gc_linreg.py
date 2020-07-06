

from statsmodels.regression.linear_model import OLS

import matplotlib.pyplot as mpl

import numpy as np

def gc_linreg(gc_data, targets, plot=False):

    """
        Performs Linear Regression on GC Activities
        Returns:  Predictions and Regression Weights
    """

    gc_pn_w_matrix = np.zeros((1000, 2))

    # fit linear model on x-coordinates
    model = OLS(targets[:, 0], gc_data).fit()

    # save weights
    gc_pn_w_matrix[:, 0] = model.params

    # fit linear model on y-coordinates
    model = OLS(targets[:, 1], gc_data).fit()

    # save weights
    gc_pn_w_matrix[:, 1] = model.params

    # save predictions
    gc_pred = np.dot(gc_data, gc_pn_w_matrix)

    if plot:
        mpl.scatter(gc_pred[:, 0], gc_pred[:, 1])
        mpl.scatter(targets[:, 0], targets[:, 1])
        mpl.show()
    return gc_pred, gc_pn_w_matrix

    
