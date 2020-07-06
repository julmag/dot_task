3 files are needed

    *_cerebellum.py # Main File
    *_calculon.py # Precalculated input for Training and Training Targets.
    circle_trajectory # Circle generation for Testing and optionally for Training if needed.

corr.py was used to correlate two gc_rates_x.npz files. Linear Regression.py to read out the reservoir with linearregression.

Update: 06.07.2020

5 files are needed:

    *_cerebellum.py # Main File
    *_calculon.py # Precalculated input for Training and Training Targets.
    circle_trajectory # Circle generation for Testing and optionally for Training if needed.
    plotter.py # plot functions used in cerebellum
    gc_linreg * linear regression function used in cerebellum
