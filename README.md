
This repository contains the python scripts developed as a part of the work presented in the paper "Extracting the Auditory Attention in a Dual-Speaker Scenario From EEG Using a Joint CNN-LSTM Model "

Paper Link: https://www.frontiersin.org/articles/10.3389/fphys.2021.700655/full

The pyTorch scripts available in this folder are:

	train.py : Scripts to train the network
  
	DNN_model.py : Neural network model
  
	import_data.py : script to import EEG and Spectrogram data from the PreProc_folder
  
	pruning_fine_tune.py : script to train the pruned model
  
	prune_utils.py : APIs used to perform magnitude pruning
  

First go through readme_pre_proc.txt in the PreProc_folder

Run the scripts in PreProc_folder to generate training and validation data. Then: 

	To train the network, run train.py. Use PreProc_folder\pre_trained_model.pth if pre-trained model is required.

	To prune a trained model, run  pruning_fine_tune.py. Set prune_percent to the required sparsity



