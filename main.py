
import time
import sys
import os
import numpy as np
import pandas as pd
import simplejson
import warnings
from signals.FeaturesExtraction import FeaturesExtraction
from classification import ClassificationSklearn as classifySklearn

binaryClassifier = None

def dict_to_json(dict):
    dict = dict.copy()
    for key, val in zip(dict.keys(), dict.values()):
        if isinstance(val, (np.ndarray, np.generic)):
             #print(type(val))
             dict[key] = val.tolist()
    return simplejson.dumps(dict)

def initClassifiers():
    global binaryClassifier
    datasetBinary = pd.read_csv(
        os.path.abspath('F:/University/Final Year/FYP/EEG/EEG-Diagnosis(Python)/data/dataset.csv'))
    dataBinary = datasetBinary.values[:, :-1]
    targetBinary = datasetBinary.values[:, -1]
    binaryClassifier = classifySklearn.ClassificationBinary(dataBinary, targetBinary)
    binaryClassifier.fitModel()
    
    
def main():
    #print(str(sys.argv))
    #print("Welcome to Python...........")
    #print('Inside Python Script....')
    
    warnings.filterwarnings("ignore");
    
    if len(sys.argv) < 5:
        raise ValueError('Not all parameters provided')

    file_path = sys.argv[1]
    sample_rate = float(sys.argv[2])
    age = sys.argv[3]
    sex = 0 if sys.argv[4] == 'm' else 1

    # print(file_path,sample_rate)
    data = pd.read_csv(file_path, header=None)
    
    feat = FeaturesExtraction(raw_eeg=data, sample_rate=sample_rate)
    results = feat.extract_features()

    initClassifiers()
    data = [age, sex, results['pfd'], results['dfa'], results['hurst_exponent'], results['theta_min'], results['theta_average'],
            results['theta_max'], results['alpha_low_min'], results['alpha_low_average'], results['alpha_low_max'], results['alpha_high_min'],
             results['alpha_high_average'], results['alpha_high_max'], results['beta_min'], results['beta_average'], results['beta_max'], 
             results['gamma_min'], results['gamma_average'], results['gamma_max'], results['sample_rate']]
    isEpilepsy = True if binaryClassifier.makePrediction(np.array(data).reshape(1, -1)) == 1 else False

    results['isEpilepsy'] = isEpilepsy
    json_content = dict_to_json(results)
    
    print(json_content)
    sys.stdout.flush()
    
    #sys.stdout.write(json_content)
    #sys.stdout.flush()


if __name__ == '__main__':
    main = main()    
