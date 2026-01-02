import mne
import pandas as pd
import numpy as np
from mne.preprocessing import ICA
from autoreject import AutoReject

# Define subject IDs and corresponding file names
subject_ids = [f'{i:02}' for i in range(1, 20)]  # Subject IDs from 01 to 19
file_suffixes = [
    'baseline_0_to_5_minutes.csv',
    'math_task_6_to_10_minutes.csv',
    'job_interview_11_to_15_minutes.csv',
    'recovery_period_16_to_20_minutes.csv'
]

# Loop over each subject and file
for subject_id in subject_ids:
    for suffix in file_suffixes:
        # Construct file path
        file_path = f'/content/drive/MyDrive/MAJID/filtered_EEG/subject_{subject_id}/subject_{subject_id}_{suffix}'
        print(f'Processing {file_path}...')  # Display the file being processed

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Assume the first 64 columns are EEG channels
        eeg_channels = df.columns[:64]

        # Rename channels to match standard montage nomenclature
        mapping = {
            'F1-0': 'F1',
            'F3-0': 'F3',
            'F5-0': 'F5',
            'F7-0': 'F7',
            'C1-0': 'C1',
            'C3-0': 'C3',
            'C5-0': 'C5',
            'F2-0': 'F2',
            'F4-0': 'F4',
            'F6-0': 'F6',
            'F8-0': 'F8',
            'C2-0': 'C2',
            'C4-0': 'C4',
            'C6-0': 'C6',
            'Afz': 'AFz'
        }
        df.rename(columns=mapping, inplace=True)
        eeg_channels = df.columns[:64]  # Update channel names after renaming
        data = df[eeg_channels].transpose().values  # Transpose to (n_channels, n_samples)

        # Create MNE info object
        sfreq = 1024  # Sampling frequency (adjust accordingly)
        info = mne.create_info(ch_names=eeg_channels.tolist(), sfreq=sfreq, ch_types='eeg')

        # Create MNE RawArray
        raw = mne.io.RawArray(data, info)

        # Set montage (standard 10-20 system)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        # Create epochs for autoreject
        epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)

        # Use AutoReject to automatically clean epochs
        ar = AutoReject()
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

        # Fit ICA on the cleaned epochs
        ica = ICA(n_components=len(eeg_channels), random_state=97, max_iter=800, method='fastica')
        ica.fit(epochs_clean)

        # Print excluded epochs
        print("Excluded epochs based on AutoReject:", reject_log.bad_epochs)

        # Automatically exclude bad components based on bad epochs
        ica.exclude = reject_log.bad_epochs

        # Print excluded components
        excluded_components = ica.exclude
        print(f'Subject {subject_id} - Excluded components:', excluded_components)

        # Apply ICA to remove the selected components from the raw data
        raw_corrected = ica.apply(raw)

        # Save the corrected data
        cleaned_file_path = f'/content/drive/MyDrive/MAJID/cleaned_EEG/subject_{subject_id}_cleaned_raw_auto.fif'
        raw_corrected.save(cleaned_file_path, overwrite=True)"
