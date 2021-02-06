Ensure that the Python environment contains all the required versions as listed in requirements.txt.

The PreProc_folder contains all the necessary scripts to preprocess the speech and the EEG signals.

Save the speech signal at 16 kHz (e.g. stim_1min.wav) and save the downsampled EEG signal at 64 Hz (e.g. eeg_1min.mat). 
The speech file should be a stereo file where each of the channel corresponds to a particular speaker in the dual-speaker auditory scene.

Run the jupyter notebook Data_processing.ipynb. This script will generate the following files and data.
	train.csv : csv file containing the filenames of the training set at 3 sec trail duration
	validation.csv : csv file containing the filenames of the validation set at 3 sec trail duration
	test.csv : csv file containing the filenames of the test set at 3 sec trail duration
	EEG : folder containing the EEG data per trial
	Speech_data : folder containing the spectrogram data per trial

At this stage, all the necessary input to the DNN are available





 




