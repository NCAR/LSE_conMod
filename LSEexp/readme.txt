exp1: 
use attributes from Xie 2021 when training lse. note that 2 attributes among Xie 2021 are missing in Yalan's data 
normNSE is used as the training target

exp2:
same with exp1, except that it uses all Yalan's basin attributes, while the below are has nan values in some basins and are not included
Missing root_depth_50 in YSong_filtered
Missing geol_porostiy in YSong_filtered
iter-0 comes from exp1

exp3:
same with exp2, but use normKGE2012 as the objective function

exp4:
same with exp2, but use truncNSE as the objective function
truncNSE means NSE<-4 is changed to -4

exp5:
same with exp2, but extend the train begin date to 1990 so I have 10 instead of 1 spin up year
iter-0 is rerun

exp6:
same with exp2, but change reduce the initial sample from 400 to 200 and reduce the trial_per_iter from 100 to 50


Note: exp1 to exp6 use Yalan' HBV model, and the camels ID can be wrong although the calibration should still be valid

exp7:
same with exp6, but use the new HBV_KM model

exp8:
same with exp7, but use following numbers (i.e., new iter-0)
num_init = 400
num_per_iter = 100
num_iter = 8

while exp7 uses
num_init = 200
num_per_iter = 50

exp9:
same with exp7, but only use 100 basins
this is almost the same with exp7 concerning median nse or kge. I start to worry that the geo-attributes are just used as basin flags to distinugish different basins. it is also as good as dPL + HBV


exp10:
same with exp8 iter-0, but reduce num_per_iter to 50 and increase num_iter to 15

exp11:
same with exp10, but increase num_per_iter to 150

exp12:
same with exp7, but uses normKGE2012 as the objective function
use iter-9 from exp7

