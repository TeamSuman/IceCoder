import os, sys
import numpy as np
import MDAnalysis as mda
import joblib
from joblib import Parallel, delayed
import torch
import tqdm.auto as tqdm
sys.path.append("../Scripts/")
from model import VAE
from soaper import soapFromUniverse


class IceCoder(VAE,soapFromUniverse):
    def __init__(self, scaler = "Saved/scaler.save", model = "Saved/pyvenc.pt", clf = "Saved/svm.save"):
        VAE.__init__(self, 952, None, None, hidden1 = 1500,
                     hidden2 = 600,hidden3=400,hidden4=100, 
                     code = 2,)
        soapFromUniverse.__init__(self)
        ## Loading previously trained model
        self.load_state_dict(torch.load(model))
        ## Loading scaler (saved)
        self.scaler = joblib.load(scaler)
        ## Loading classifier (SVM)
        self.clf = joblib.load(clf)
        
    def featurizer(self, mode = "aggressive", mda_universe = None, start = 0, stop = -1, step = 1, verbose = False):
        results = None
        self.start = start
        self.stop = stop
        self.step = step
        if mode == "aggressive":
            results = Parallel(n_jobs=-1)(delayed(self.soaper)(mda_universe, frame) for frame in tqdm.trange(start, stop, step))
        elif mode == "serial":
            results = []
            if verbose:
                for frame in tqdm.trange(start, stop, step):
                    results.append(self.soaper(mda_universe, frame))
            else:
                for frame in range(start, stop, step):
                    results.append(self.soaper(mda_universe, frame))
        elif mode == "efficient":
            results = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.soaper)(mda_universe, frame) for frame in tqdm.trange(start, stop, step))
        else:
            raise ValueError("Something wrong happenned!")
            
        soaps = np.concatenate(results, dtype = np.float32)
        self.transformed = self.scaler.transform(soaps)
        
    def project(self):
        mus = self.encode(torch.Tensor(self.transformed))[1].detach().to('cpu').numpy()
        self.projected = mus
    def predict(self):
        if not hasattr(self, 'projected'):
            self.project()
        self.classified = self.clf.predict(self.projected)
        self.ices = list(map(self.mapintoice, self.classified))
        
    def mapintoice(self, x):
        mapdict = {0 : "Liquid",
                   1 : "Ice-Ih",
                   2 : "Ice-Ic", 
                   3 : "Ice-II", 
                   4 : "Ice-III",
                   5 : "Ice-V",
                   6 : "Ice-XVII",
                   7 : "Ice-VI", 
                   8 : "Ice-VII"}
        return mapdict[x]
            