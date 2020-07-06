3 files are needed

    *_cerebellum.py # Main File
    *_calculon.py # Precalculated input for Training and Training Targets.
    circle_trajectory # Circle generation for Testing and optionally for Training if needed.

corr.py was used to correlate two gc_rates_x.npz files. Linear Regression.py to read out the reservoir with linearregression.

Update: 06.07.2020

6 files are needed:

    1. *_cerebellum.py     # Main File
    2. *_calculon.py       # Precalculated input for Training and Training Targets.
    c3. ircle_trajectory   # Circle generation for Testing and optionally for Training if needed.
    4. plotter.py          # plot functions used in cerebellum
    5. gc_linreg           # linear regression function used in cerebellum
    6. gc_matrix           # weight matrix for the gc_pn connection, will be overritten by running in the firt epoch, but is need to run at least once. 
