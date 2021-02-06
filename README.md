
First go through readme_pre_proc.txt in the PreProc_folder folder 

The important pyTorch scripts in this folder are:

	train.py : Scripts to train the network
  
	DNN_model.py : Neural network model
  
	import_data.py : script to import EEG and Spectrogram data from the PreProc_folder
  
	pruning_fine_tune.py : script to train the pruned model
  
	prune_utils.py : APIs used to perform magnitude pruning
  
  

To train the network, run train.py. Use PreProc_folder\pre_trained_model.pth if pre-trained model is required.

To prune a trained model, run  pruning_fine_tune.py. Set prune_percent to the required sparsity



