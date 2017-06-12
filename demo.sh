cd code
python competitors.py --method 1 --A_type load --x_type ctm
python AND.py --A_type load --x_type ctm --ctm_sigma 1.0 --noise 0.0 --init_thres 0.1 --thres_decay 1.1 --outer_epo 200 
