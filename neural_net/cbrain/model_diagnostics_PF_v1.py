"""
Contains the ModelDiagnostics class.
TODO:
- Colorbar
- Axes labels
"""

# Imports
from .imports import *
from .data_generator import DataGenerator
from .preprocess_aqua import L_V, C_P, conversion_dict
import pickle
import keras
import os

def moving_average(a, n=12) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def wtd_cal(yi,xi,res,dz_scale,nz,dz):
    satur_arr = res[:,50:,yi,xi]
    press_arr = res[:,:50,yi,xi]
    satur_idx = np.sum(satur_arr>=1,axis=1)-1
    if dz_scale is None:
        cell_depth = nz*dz - dz*(satur_idx+1)+dz/2.
    else:
        total_depth = np.sum(dz*np.array(list(dz_scale.values())))
        depth_dict = {}
        depth_dict[list(dz_scale.keys())[0]] = nz*dz - 0.5*(list(dz_scale.values())[0]*dz)
        for ii,ki in enumerate(list(dz_scale.keys())[1:]):
            depth_dict[ki]=(total_depth - np.sum(dz*np.array(list(dz_scale.values())[:ii+1]))-dz_scale[ki]*dz/2)
        max_wtd = np.vectorize(depth_dict.get)(satur_idx)
        cell_depth = max_wtd.astype('float')
    press_sel = press_arr[:,satur_idx]
    wtd = cell_depth -press_sel[:,0]
    return wtd

class ModelDiagnostics(object):
    """
    Two basic functionalities:
    1. Plotting --> need preds and truth of selected time step in original values for one var
    2. Global statistics --> also from denormalized values

    Differences between TF and Keras:
    1. Data loading: For Keras I will use my data_generator (much faster),
                     for TF I will read and process the raw aqua files
    2. Output normalization
    3. Output shape: 1D for Keras, 2D for TF --> Use TF convention
    NOTE: This cannot handle outputs with one level.
    """
    def __init__(self, predictions, targets, target_names, tvars, is_tf=False):
        # Basic setup
        self.is_tf = is_tf; self.is_k = not is_tf
        # Get variable names and open arrays
        if self.is_k:
            self.tvars = tvars
            self.target_names = target_names
            self.sample_size,self.time,self.target_size,self.nlat,self.nlon = predictions[0].shape
            self.batch_size = self.nlat*self.nlon
            p_s = []
            for p in predictions:
                p_s.append(p[0,:,:,:,:])
            self.p = p_s
            self.t = targets[0,:,:,:,:]
        else:
            print('Please check prediction and target objects!')

    # Init helper functions    
    def get_pt(self, itime, var=None):
        """
        Returns denormalized predictions and truth for a given time step and var.
        [lat, lon, lev] or [lat, lon, var, lev] if var is None
        """
        p, t = self._get_k_pt(itime, var)
        return p, t

    def _get_k_pt(self, itime, var=None):
        """Keras version"""
        p_ss = []
        for p in self.p:
            p_ss.append(p[itime,:,:,:])
        t = self.t[itime,:,:,:]
        # At this stage they have shape [ngeo, stacked_levs]
        return p_ss,t
    
    def _get_var_idxs(self, var, cutoff=0):
        idxs = np.array([i for i, n in enumerate(self.target_names) if var in n])
        if not idxs.size == 1: idxs = idxs[cutoff:]
        return idxs

    # Plotting functions
    def plot_double_xy(self, itime, ilev, var, **kwargs):
        p, t = self.get_pt(itime, var)
        if t.ndim == 3: 
            t = t[ilev,:, :]
            p_ss = []
            for pi in p:
                p_ss.append(pi[ilev,:,:])
        return self.plot_double_slice(p_ss, t, **kwargs)
    
    def plot_hydrograph(self,yi,xi,ilev, mv=12, **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        colors = ['g', 'r','c','m','y','k']
        for ci,p in enumerate(self.p):
            predict_graph = p[:,ilev,yi,xi]
            predict_mean = moving_average(predict_graph,mv)
            ax.plot(predict_mean,colors[ci],label='Prediction #'+str(ci))
        true_graph = self.t[:,ilev,yi,xi]
        true_mean = moving_average(true_graph,mv)
        ax.plot(true_mean,'b',label='Truth')
        ax.legend()
        return fig, ax
    
    def plot_wtd(self,yi,xi,dz_scale,nz,dz,mv=1,**kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        colors = ['g', 'r','c','m','y','k']
        true_wtd = wtd_cal(yi,xi,self.t,dz_scale,nz,dz)
        true_wtd_mean = moving_average(true_wtd,mv)
        ax.plot(true_wtd_mean,'b',label='Truth')
        for ci,p in enumerate(self.p):
            pred_wtd = wtd_cal(yi,xi,p,dz_scale,nz,dz)
            pred_wtd_mean = moving_average(pred_wtd,mv)
            ax.plot(pred_wtd_mean,colors[ci],label='Prediction #'+str(ci))
        ax.legend()
        return fig, ax

    def plot_double_yz(self, itime, ilon, var, **kwargs):
        p, t = self.get_pt(itime, var)
        return self.plot_double_slice([x[:, :,ilon].T for x in p], t[:, :, ilon].T, **kwargs)

    def plot_double_slice(self, p, t, title='', unit='', **kwargs):
        fig, axes = plt.subplots(1, len(p)+1, figsize=(12, 5))
        I1 = axes[0].imshow(t, **kwargs)
        axes[0].set_title('Truth')
        for axi,pi in enumerate(p):
            I2 = axes[axi+1].imshow(pi, **kwargs)
            axes[axi+1].set_title('Prediction #'+str(axi+1))
            cb2 = fig.colorbar(I2, ax=axes[axi+1], orientation='horizontal')
            cb2.set_label(unit)
        cb1 = fig.colorbar(I1, ax=axes[0], orientation='horizontal')
        cb1.set_label(unit); 
        fig.suptitle(title)
        return fig, axes

    def plot_slice(self, x, title='', unit='', **kwargs):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        I = ax.imshow(x, **kwargs)
        cb = fig.colorbar(I, ax=ax, orientation='horizontal')
        cb.set_label(unit)
        ax.set_title(title)
        return fig

    # Statistics computation
    def compute_stats(self, niter=None):
        """Compute statistics in for [lat, lon, var, lev]"""
        nt = self.sample_size // self.batch_size
        if niter is not None: nt = niter
        # Allocate stats arrays
        psum = np.zeros((self.nlat, self.nlon, self.target_size))
        tsum = np.copy(psum); sse = np.copy(psum)
        psqsum = np.copy(psum); tsqsum = np.copy(psum)
        for itime in tqdm(range(nt)):
            if self.is_k:
                p, t = self.get_pt(itime)   # [lat, lon, var, lev]
                p = p[0]
            else:   # For TF load entire aqua file at once!
                itmp = itime % self.ntime; idate = itime // self.ntime
                if itmp == 0:
                    pday, tday = self._get_tf_pt(idate=idate)
                p, t = (pday[itmp], tday[itmp])
            # Compute statistics
            psum += p; tsum += t
            psqsum += p ** 2; tsqsum += t ** 2
            sse += (t - p) ** 2
        # Compute average statistics
        self.stats = {}
        pmean = psum / nt; tmean = tsum / nt
        self.stats['bias'] = pmean - tmean
        self.stats['mse'] = sse / nt
        self.stats['pred_mean'] = psum / nt
        self.stats['true_mean'] = tsum / nt
        self.stats['pred_sqmean'] = psqsum / nt
        self.stats['true_sqmean'] = tsqsum / nt
        self.stats['pred_var'] = psqsum / nt - pmean ** 2
        self.stats['true_var'] = tsqsum / nt - tmean ** 2
        self.stats['r2'] = 1. - (self.stats['mse'] / self.stats['true_var'])
        # Compute horizontal stats [var, lev]
        self.stats['hor_tsqmean'] = np.mean(self.stats['true_sqmean'], axis=(0,1))
        self.stats['hor_tmean'] = np.mean(self.stats['true_mean'], axis=(0, 1))
        self.stats['hor_mse'] = np.mean(self.stats['mse'], axis=(0, 1))
        self.stats['hor_tvar'] = self.stats['hor_tsqmean'] - self.stats['hor_tmean'] ** 2
        self.stats['hor_r2'] = 1 - (self.stats['hor_mse'] / self.stats['hor_tvar'])

    def mean_stats(self, cutoff_level=0):
        """Get average statistics for each variable and returns dataframe"""
        df = pd.DataFrame(index=self.tvars + ['all'],
            columns=list(self.stats.keys()))
        for ivar, var in enumerate(self.tvars):
            for stat_name, stat in self.stats.items():
                # Stats have shape [lat, lon, var, lev]
                df.loc[var, stat_name] = np.mean(stat[..., self._get_var_idxs(var, cutoff_level)])
        df.loc['all']['hor_r2'] = np.mean(df['hor_r2'].mean())
        self.stats_df = df
        return df

    def save_stats(self, path=None):
        if path is None:
            os.makedirs('./tmp', exist_ok=True)
            path= './tmp/' + self.save_str
        with open(path, 'wb') as f: pickle.dump(self.stats, f)

    def load_stats(self, path=None):
        if path is None: path= './tmp/' + self.save_str
        with open(path, 'rb') as f: self.stats = pickle.load(f)





    # def _compute_SPDT_SPDQ(self, f, t, p):
    #     # Get dP
    #     dP = self._get_dP(f)
    #     SPDT_pred = self.vint((p[:, self._get_var_idxs('target', 'SPDT')]),
    #                           C_P, dP)
    #     SPDQ_pred = self.vint((p[:, self._get_var_idxs('target', 'SPDQ')]),
    #                           L_V, dP)
    #     SPDT_true = self.vint((t[:, self._get_var_idxs('target', 'SPDT')]),
    #                           C_P, dP)
    #     SPDQ_true = self.vint((t[:, self._get_var_idxs('target', 'SPDQ')]),
    #                           L_V, dP)
    #     return np.square(SPDT_pred + SPDQ_pred), np.square(SPDT_true + SPDQ_true)
    #
    # @staticmethod
    # def vint(x, factor, dP):
    #     return np.sum(x * factor * dP / G, -1)
