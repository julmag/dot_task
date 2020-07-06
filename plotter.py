#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:09:39 2020

@author: jthukral
"""
import numpy as np
import matplotlib.pyplot as plt
from gc_linreg import *
import matplotlib as mpl
from datetime import datetime

def plot_prediction_over_trials(gc_rates_end_of_trial, weights, targets):  
    plt.figure(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))
    weights = np.transpose(weights)
    plt.subplot(211)
    plt.title("Weighted Final GC Responses over Trials")
    plt.plot(gc_rates_end_of_trial * weights[:,0])
    plt.plot(targets[:,0], color='red', lw=3, label="X-Targets")
    plt.plot(np.sum(gc_rates_end_of_trial * weights[:,0], 1), color='black', lw=3, alpha=0.4, label="Predictions (sum of weighted GC)")
    plt.legend(bbox_to_anchor=(0.6, 0.2), loc=2, borderaxespad=0.)
    plt.subplot(212)
    plt.plot(gc_rates_end_of_trial * weights[:,1])
    plt.plot(targets[:,1], color='red', lw=3, label="Y-Targets")
    plt.plot(np.sum(gc_rates_end_of_trial * weights[:,1], 1), color='black', lw=3, alpha=0.4, label="Predictions  (sum of weighted GC)")
    plt.legend(bbox_to_anchor=(0.6, 0.2), loc=2, borderaxespad=0.)
    plt.show()
    
    
def plot_prediction_within_trial(gc_rates_within_trial, weights, targets, trial_no, trial_length=75, start=10):   
    plt.figure(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))
    weights = np.transpose(weights)
    plt.subplot(211)
    plt.title("Weighted GC Rates within Trial "+str(trial_no))
    plt.plot((gc_rates_within_trial*weights[:,0])[start:])
    plt.plot((targets[trial_no, 0],)*(trial_length - start), color='red', lw=3, label="X-Target")
    plt.plot(np.sum(gc_rates_within_trial * weights[:,0], 1)[start:], color='black', lw=3, alpha=0.4, label="Sum of Weighted GC rates")
    plt.legend(bbox_to_anchor=(0.05, 0.2), loc=2, borderaxespad=0.)
    plt.subplot(212)
    plt.plot((gc_rates_within_trial*weights[:,1])[start:])
    plt.plot((targets[trial_no, 1],)*(trial_length - start), color='red', lw=3, label="Y-Target")
    plt.plot(np.sum(gc_rates_within_trial * weights[:,1], 1)[start:], color='black', lw=3, alpha=0.4, label="Sum of Weighted GC rates")
    plt.legend(bbox_to_anchor=(0.05, 0.2), loc=2, borderaxespad=0.)
    plt.show()
    
    
def gc_activtiy_linreg_fit_target (title, gc_response_list, targets):
    plt.figure(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))
    gc_linreger, gc_matrix = gc_linreg(gc_response_list,targets)
    plt.scatter(gc_linreger[:,0], gc_linreger[:,1], s=0.7, c='black')
    plt.xlabel("X-Coordiante")
    plt.ylabel("Y-Coordinate")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box') 
    
    
def mse_plot (title, targets, predictions, errors):   
    
    red=np.zeros((1,4))
    blue=np.zeros((1,4))
    
    # Since scatterplots don't have the attribute alpha(transperency). Alpha is set in the definition of the color, --> last number in []=alpha
    red[0,:]=[1,0,0,0.3] 
    blue[0,:]=[0,0,1,0.3]
    
    plt.figure(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))
    mpl.rcParams['figure.figsize'] = [10, 10]
    plt.scatter(targets[:, 0], targets[:, 1], c='black', s=0.1 , label="Targets")
    plt.scatter(predictions[:, 0], predictions[:, 1],  c=np.squeeze(errors) , s=0.1, label="Predictions")    
    plt.colorbar(label='Euclidean Distance of Prediction to Target' )
    plt.legend(shadow=True, fontsize='x-large', markerscale=20, loc='upper center')
    plt.xlabel("X-Coordiante")
    plt.ylabel("Y-Coordinate")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    
    
    
    
    
    
    
    

'''Figure with Mean, max and min of TRAINING Granule Cell Activity over time. X-Axis = number_epochs * number_training examples 
needs correct input. This is just a backup for the plot'''
# gc1_figure = plt.figure(2)
# mpl.rcParams['figure.figsize'] = [10,10]
# plt.title("Granule Cell Training Activity - Blue: mean, Green: min/max")
# plt.plot(gc_response_1_list[:,0], color='b')
# plt.fill_between(range(len(gc_response_1_list[:,1])),y1=gc_response_1_list[:,1], y2=gc_response_1_list[:,2], color='g', alpha=0.5)
# plt.xlabel("Time - number_epochs * number_test_examples")
# plt.ylabel("Activity")
# plt.gca().set_aspect('equal', adjustable='box')
#     