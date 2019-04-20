
import numpy as np
import pandas as pd
import mne
import biosppy
import pyeeg


class FeaturesExtraction(object):
    raw_eeg = None
    filtered_signal = None
    sample_rate = None
    mean = None
    first_order_diff = None
    pfd = None
    dfa = None
    hurst_exponent = None
    theta = None
    theta_average = None
    theta_max = None
    theta_min = None
    alpha_low = None
    alpha_low_average = None
    alpha_low_max = None
    alpha_low_min = None
    alpha_high = None
    alpha_high_average = None
    alpha_high_max = None
    alpha_high_min = None
    beta = None
    beta_average = None
    beta_max = None
    beta_min = None
    gamma = None
    gamma_average = None
    gamma_max = None
    gamma_min = None
    
    eeg_features = {}

    def __init__(self, raw_eeg, sample_rate):
        self.raw_eeg_return = raw_eeg[0].values
        self.raw_eeg = np.array(raw_eeg)
        #print(self.raw_eeg.shape)
        #print(self.raw_eeg)
        self.sample_rate = sample_rate

    def extract_features(self):
        self.impute_values()
        self.signal_filteration()
        self.calc_mean()
        self.calc_first_order_diff()
        self.find_pfd()
        self.find_dfa()
        self.find_hurst_exponent()
        self.get_power_features()


        self.eeg_features['raw_eeg'] = np.round_(self.raw_eeg_return, decimals=2)
        self.eeg_features['filter_signal'] = np.round_(self.filtered_signal_returned[0].values, decimals=2)

        self.eeg_features['pfd'] = np.round(self.pfd, decimals=3)
        self.eeg_features['dfa'] = np.round(self.dfa, decimals=3)
        self.eeg_features['hurst_exponent'] = np.round(self.hurst_exponent, decimals=3)
        
        self.eeg_features['theta'] = np.round(self.theta, decimals=3)
        self.eeg_features['theta_min'] = np.round(self.theta_min, decimals=3)
        self.eeg_features['theta_average'] = np.round(self.theta_average, decimals=3)
        self.eeg_features['theta_max'] = np.round(self.theta_max, decimals=3)
        
        self.eeg_features['alpha_low'] = np.round(self.alpha_low, decimals=3)
        self.eeg_features['alpha_low_min'] = np.round(self.alpha_low_min, decimals=3)
        self.eeg_features['alpha_low_average'] = np.round(self.alpha_low_average, decimals=3)
        self.eeg_features['alpha_low_max'] = np.round(self.alpha_low_max, decimals=3)
        
        self.eeg_features['alpha_high'] = np.round(self.alpha_high, decimals=3)
        self.eeg_features['alpha_high_min'] = np.round(self.alpha_high_min, decimals=3)
        self.eeg_features['alpha_high_average'] = np.round(self.alpha_high_average, decimals=3)
        self.eeg_features['alpha_high_max'] = np.round(self.alpha_high_max, decimals=3)
        
        self.eeg_features['beta'] = np.round(self.beta, decimals=3)
        self.eeg_features['beta_min'] = np.round(self.beta_min, decimals=3)
        self.eeg_features['beta_average'] = np.round(self.beta_average, decimals=3)
        self.eeg_features['beta_max'] = np.round(self.beta_max, decimals=3)
        
        self.eeg_features['gamma'] = np.round(self.gamma, decimals=3)
        self.eeg_features['gamma_min'] = np.round(self.gamma_min, decimals=3)
        self.eeg_features['gamma_average'] = np.round(self.gamma_average, decimals=3)
        self.eeg_features['gamma_max'] = np.round(self.gamma_max, decimals=3)
        
        self.eeg_features['sample_rate'] = self.sample_rate

        return self.eeg_features
    
    def impute_values(self):
        x = self.raw_eeg
        x_mean = np.mean(x[x!=0])
        
        for i in range(len(x)):
            if(x[i]==0):
                x[i]=x_mean
                
        self.raw_eeg = x
    
    
    def signal_filteration(self):
        ch_names = ['EEG1']
        ch_types = ['eeg']
        sfreq = self.sample_rate
        
        info = mne.create_info(ch_names, sfreq, ch_types)
        custom_raw = mne.io.RawArray(self.raw_eeg.T, info)
        
        custom_raw.filter(None, 50, fir_design='firwin')
        
        self.filtered_signal = custom_raw.get_data()
        self.filtered_signal = self.filtered_signal.T
        
        self.filtered_signal_returned = pd.DataFrame(data=self.filtered_signal)
    

    def calc_mean(self):
        self.mean = np.mean(self.filtered_signal)
    
    def calc_first_order_diff(self):
        x = self.filtered_signal
        y = np.zeros(x.shape)
        
        for i in range(1, len(x)):
            y[1:] = x[1:] - x[:-1]
            
        self.first_order_diff = y
    
    def find_pfd(self):
        self.pfd = pyeeg.pfd(self.filtered_signal, self.first_order_diff)
        

    def find_dfa(self):
        self.dfa = pyeeg.dfa(self.filtered_signal, self.mean)
        
    def find_hurst_exponent(self):
        self.hurst_exponent = pyeeg.hurst(self.filtered_signal)
        
    def get_power_features(self):
        out = biosppy.signals.eeg.eeg(self.filtered_signal, self.sample_rate, show=False)
        
        self.theta = np.array(out['theta'])
        self.theta_average = np.mean(self.theta)
        self.theta_max = np.max(self.theta)
        self.theta_min = np.min(self.theta)
        
        self.alpha_low = np.array(out['alpha_low'])
        self.alpha_low_average = np.mean(self.alpha_low)
        self.alpha_low_max = np.max(self.alpha_low)
        self.alpha_low_min = np.min(self.alpha_low)
        
        self.alpha_high = np.array(out['alpha_high'])
        self.alpha_high_average = np.mean(self.alpha_high)
        self.alpha_high_max = np.max(self.alpha_high)
        self.alpha_high_min = np.min(self.alpha_high)
        
        self.beta = np.array(out['beta'])
        self.beta_average = np.mean(self.beta)
        self.beta_max = np.max(self.beta)
        self.beta_min = np.min(self.beta)
        
        self.gamma = np.array(out['gamma'])
        self.gamma_average = np.mean(self.gamma)
        self.gamma_max = np.max(self.gamma)
        self.gamma_min = np.min(self.gamma)
        