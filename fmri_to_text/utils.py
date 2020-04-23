""" Regroup functions to create (or load) a masker from group of subjects/runs 
and extract specific list of scans.

First, specify the logs_path (after the imports).
Second, you need to create a yaml file like the following template:

###################
# Yaml template
masker_path: /home/td******
language: english
tr: 2.0
nscans: # values for english
    - 282
    - 298
    - 340
    - 303
    - 265
    - 343
    - 325
    - 292
    - 368
shift: 10 # nb of seconds to take into account after the end of the sentence
path_to_fmridata: /neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/
path_to_fMRI_example: /neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/fMRI/english/sub-057/func/fMRI_*run*
path_to_onset_file_example:  /neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/wave/english/onsets-offsets/word+punctuation_run*
###################

At the end of the script (in the ‘if __name__ ...‘ section), I wrote an example on how to use 
the functions.
You have two options:
    - To include them during your model training if reading fMRI with masker and extracting 
    scans is not too long
    - Read fMRI data with the masker and saving it somewhere to only have to extract the
    information when training without having to transform fMRI data.
To decide, just check on one example, and look at the log file where I included the time taken 
by each function.
"""

import os
import yaml
import time
import logging
import argparse
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.image import math_img, mean_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_glass_brain, plot_img

logs_path = './logs.txt'
logging.basicConfig(filename=logs_path,level=logging.INFO)

def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass
    
def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
            quit()
    return parameters

def fetch_data(path_to_fmridata, subject, language, models=[]):
    """ Retrieve fmri data.
    Arguments:
        - path_to_fmridata: str (path to folder containing subjects folders)
        - subject: str
        - language: str
        - models: list (of dict)
    """
    fmri_path = os.path.join(path_to_fmridata, "fMRI", language, subject, "func")
    fMRI_paths = sorted(glob.glob(os.path.join(fmri_path, 'fMRI_*run*')))
    return fMRI_paths

def compute_global_masker(files, smoothing_fwhm=None): # [[path, path2], [path3, path4]]
    """Returns a MultiNiftiMasker object from list (of list) of files.
    Arguments:
        - files: list (of list of str)
    Returns:
        - masker: MultiNiftiMasker
    """
    masks = [compute_epi_mask(f) for f in files]
    global_mask = math_img('img>0.5', img=mean_img(masks)) # take the average mask and threshold at 0.5
    masker = MultiNiftiMasker(global_mask, detrend=True, standardize=True, smoothing_fwhm=smoothing_fwhm)
    masker.fit()
    return masker

def fetch_masker(masker_path, language, path_to_fmridata, smoothing_fwhm=None):
    """ Fetch or compute if needed a global masker from all subjects of a
    given language.
    Arguments:
        - masker_path: str
        - language: str
        - path_to_fmridata: str
        - smoothing_fwhm: int
    """
    t0 = time.time()
    if os.path.exists(masker_path + '.nii.gz') and os.path.exists(masker_path + '.yml'):
        logging.info(" loading existing masker...") 
        params = read_yaml(masker_path + '.yml')
        mask_img = nib.load(masker_path + '.nii.gz')
        masker = MultiNiftiMasker()
        masker.set_params(**params)
        masker.fit([mask_img])
    else:
        logging.info(" recomputing masker...") 
        fmri_runs = {}
        subjects = [get_subject_name(id) for id in possible_subjects_id(language)]
        for subject in subjects:
            fmri_paths = fetch_data(path_to_fmridata, subject, language)
            fmri_runs[subject] = fmri_paths
        masker = compute_global_masker(list(fmri_runs.values()), smoothing_fwhm=smoothing_fwhm)
        params = masker.get_params()
        params = {key: params[key] for key in ['detrend', 'dtype', 'high_pass', 'low_pass', 'mask_strategy', 
                                                'memory_level', 'n_jobs', 'smoothing_fwhm', 'standardize',
                                                't_r', 'verbose']}
        nib.save(masker.mask_img_, masker_path + '.nii.gz')
        save_yaml(params, masker_path + '.yml')
    logging.info('\tDone in: {}'.format(time.time() - t0))
    return masker

def read_fmri_data(masker, path_to_raw_scan):
    """ Returns numpy array from fmri data.
    Arguments: 
        - masker: MultiNiftiMasker
        - path_to_raw_scan: str
    """
    t0 = time.time()
    logging.info(" Reading fMRI data...") 
    data = masker.transform(path_to_raw_scans)
    logging.info('\tDone in: {}'.format(time.time() - t0))
    return data

def get_scans(data, onsets_list, tr, n_scans, shift=10):
    """ Returns a list of numpy arrays from fmri data np.array and a list of
    onsets-offsets.
    Arguments:
        - data: str or np.array
        - onsets_list: list of tuple (e.g.: [(onset_0, offset_0), ..., (onset_n, offset_n)])
        - tr: float (sampling rate)
        - n_scans: int
        - shift: float/int (time shift to take into account after the end of the sentence)
    Returns:
        - result: np.array (dim: (len(onsets_list), end_row - beg_row + 1, nb_voxels))
    """
    t0 = time.time()
    logging.info(" Retrieving scans...") 
    if type(path_to_array)==str:
        data = np.load(data)
    result = []
    max_row = len(data) - 1
    for onset, offset in onsets_list:
        beg_row = onset // tr
        end_row = min((offset + shift) // tr, max_row)
        result.append(data[beg_row : end_row + 1, :])
    result = np.stack(result, axis=0)
    logging.info('\tDone in: {}'.format(time.time() - t0))
    return result

def textgrid_to_sentence(onset_df):
    """Split textgrid dataframe into list of tuple (sentence_onset, sentence_offset).
    Arguments:
        - onset_df: pd.DataFrame
    """
    result = []
    eos_punctuation = [".", "?", "!", "..."]
    beg_row = 0
    last_index = len(onset_df) - 1
    for index, row in onset_df.iterrows():
        if row['word'] in eos_punctuation:
            result.append((onset_df.iloc[beg_row]['onsets'], onset_df.iloc[index]['offsets']))
            beg_row = min(index, last_index)
    return result



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Create a masker from group of subjects/runs and extract specific list of scans.")
    parser.add_argument('--yaml_file', type=str, help="Path to the yaml file containing needed information.")
    args = parser.parse_args()

    parameters = read_yaml(args.yaml_file)
    nscans = parameters['nscans'][0] # nb of scans of the first run for english scans
    tr = parameters['tr']
    shift = parameters['shift']

    masker = fetch_masker(parameters['masker_path'], parameters['language'], parameters['path_to_fmridata'], input_path)

    fMRI_example = sorted(glob.glob(parameters['path_to_fMRI_example']))[0] # get 1 scan of 1 subject to test functions
    onset_example = sorted(glob.glob(parameters['path_to_onset_file_example']))[0] # get onset-offset file for the fMRI example
    data_array = read_fmri_data(masker, path_to_raw_scan) # transform FMRI data to 2D array
    onset_offset = pd.read_csv(onset_example)
    onset_offset_list = textgrid_to_sentence(onset_offset)
    list_of_scans = get_scans(data_array, onset_offset_list, tr, n_scans, shift=shift)
