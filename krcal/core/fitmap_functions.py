"""Module fitmap_functions.
This module performs lifetime fits to DataFrame maps

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https

Author: JJGC
Last revised: Feb, 2019

"""
import numpy as np
import warnings
from   pandas               import DataFrame

from typing                 import List
from typing                 import Tuple
from typing                 import Dict

from . core_functions       import value_from_measurement
from . core_functions       import uncertainty_from_measurement
from . fit_lt_functions     import fit_lifetime
from . fit_lt_functions     import pars_from_fcs
from . selection_functions  import get_time_series_df
from . kr_types             import FitType, FitParTS


import logging
log = logging.getLogger(__name__)


def time_fcs_df(ts      : np.array,
                masks   : List[np.array],
                dst     : DataFrame,
                nbins_z : int,
                nbins_e : int,
                range_z : Tuple[float, float],
                range_e : Tuple[float, float],
                energy  : str                 = 'S2e',
                z       : str                 = 'Z',
                fit     : FitType             = FitType.unbined)->FitParTS:
    """
    Fit lifetime of a time series.

    Parameters
    ----------
        ts
            A vector of floats with the (central) values of the time series.
        masks
            A list of boolean vectors specifying the selection masks that define the time series.
        dst
            A dst DataFrame
        range_z
            Range in Z for fit.
        nbins_z
            Number of bins in Z for the fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        energy:
            Takes by default S2e (uses S2e field in dst) but can take any value specified by str.
        z:
            Takes by default Z (uses Z field in dst) but can take any value specified by str.
        fit
            Selects fit type.


    Returns
    -------
        A FitParTs:

    @dataclass
    class FitParTS:             # Fit parameters Time Series
        ts   : np.array         # contains the time series (integers expressing time differences)
        e0   : np.array         # e0 fitted in time series
        lt   : np.array         # lt fitted in time series
        c2   : np.array         # c2 fitted in time series
        e0u  : np.array         # e0 error fitted in time series
        ltu  : np.array

    """

    dsts = [dst[sel_mask] for sel_mask in masks]

    logging.debug(f' time_fcs_df: len(dsts) = {len(dsts)}')
    fcs =[fit_lifetime(dst[z].values, dst[energy].values,
                       nbins_z, nbins_e, range_z, range_e, fit) for dst in dsts]

    e0s, lts, c2s = pars_from_fcs(fcs)

    return FitParTS(ts  = np.array(ts),
                    e0  = value_from_measurement(e0s),
                    lt  = value_from_measurement(lts),
                    c2  = c2s,
                    e0u = uncertainty_from_measurement(e0s),
                    ltu = uncertainty_from_measurement(lts))


def fit_map_xy_df(selection_map : Dict[int, List[DataFrame]],
                  event_map     : DataFrame,
                  n_time_bins   : int,
                  time_diffs    : np.array,
                  nbins_z       : int,
                  nbins_e       : int,
                  range_z       : Tuple[float, float],
                  range_e       : Tuple[float, float],
                  energy        : str                 = 'S2e',
                  z             : str                 = 'Z',
                  fit           : FitType             = FitType.profile,
                  n_min         : int                 = 100)->Dict[int, List[FitParTS]]:
    """
    Produce a XY map of fits (in time series).

    Parameters
    ----------
        selection_map
            A DataFrameMap of selections, defining a selection of events.
        event_map
            A DataFrame, containing the events in each XY bin.
        n_time_bins
            Number of time bins for the time series.
        time_diffs
            Vector of time differences for the time series.
        nbins_z
            Number of bins in Z for the fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        energy:
            Takes by default S2e (uses S2e field in dst) but can take any value specified by str.
        z:
            Takes by default Z (uses Z field in dst) but can take any value specified by str.
        fit
            Selects fit type.
        n_min
            Minimum number of events for fit.


    Returns
    -------
        A Dict[int, List[FitParTS]]
        @dataclass
        class FitParTS:             # Fit parameters Time Series
            ts   : np.array          # contains the time series (integers expressing time differences)
            e0   : np.array          # e0 fitted in time series
            lt   : np.array
            c2   : np.array
            e0u  : np.array          # e0 error fitted in time series
            ltu  : np.array

    """

    def fit_fcs_in_xy_bin (xybin         : Tuple[int, int],
                           selection_map : Dict[int, List[DataFrame]],
                           event_map     : DataFrame,
                           n_time_bins   : int,
                           time_diffs    : np.array,
                           nbins_z       : int,
                           nbins_e       : int,
                           range_z       : Tuple[float, float],
                           range_e       : Tuple[float, float],
                           energy        : str                 = 'S2e',
                           z             : str                 = 'Z',
                           fit           : FitType             = FitType.profile,
                           n_min         : int                 = 100)->FitParTS:
        """Returns fits in the bin specified by xybin"""

        i = xybin[0]
        j = xybin[1]
        nevt = event_map[i][j]
        tlast = time_diffs.max()
        tfrst = time_diffs.min()
        ts, masks =  get_time_series_df(n_time_bins, (tfrst, tlast), selection_map[i][j])

        logging.debug(f' ****fit_fcs_in_xy_bin: bins ={i,j}')

        if nevt > n_min:
            logging.debug(f' events in fit ={nevt}, time series = {ts}')
            return time_fcs_df(ts, masks, selection_map[i][j],
                               nbins_z, nbins_e, range_z, range_e, energy, z, fit)
        else:
            warnings.warn(f'Cannot fit: events in bin[{i}][{j}] ={event_map[i][j]} < {n_min}',
                         UserWarning)

            dum = np.zeros(len(ts), dtype=float)
            dum.fill(np.nan)
            return FitParTS(ts, dum, dum, dum, dum, dum)

    logging.debug(f' fit_map_xy_df')
    fMAP = {}
    r, c = event_map.shape
    for i in range(r):
        fMAP[i] = [fit_fcs_in_xy_bin((i,j), selection_map, event_map, n_time_bins, time_diffs,
                                     nbins_z, nbins_e, range_z,range_e, energy, z, fit, n_min)
                                     for j in range(c) ]
        
    x = [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    y = [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    yu = [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    
    print('Number of xy bins:',len(selection_map))
    print('Fit type:',str(fit).replace('FitType.',''))
    print('Nmin:',n_min)

#     for k in range(0,len(selection_map),1):
#         for l in range(0,len(selection_map),1):
#             if event_map[k][l] > 1:
#                 r_pos = ((((55*2)/50)*(k - 24))**2 + ((((55*2)/50)*(l - 24))**2))**(1/2)

#                 fig = plt.figure();
#                 x[k][l], y[k][l], yu[k][l] = profileX(selection_map[k][l].Z, selection_map[k][l].S2e, 30, (0, 500));
#                 plt.figure(figsize=(18,8))
#                 plt.subplot(1,2,1)
#                 plt.errorbar(x[k][l],y[k][l],yu[k][l],fmt="kp");
#                 plt.plot(x[k][l],fMAP[k][l].e0*np.exp(-x[k][l]/fMAP[k][l].lt),'r',(15,320));
#                 textstr_1 = 'Bins: {:.0f} (nmin = {:.0f})\nFitType: {} ({:.0f} evts.)\nPosition xy: ({:.0f},{:.0f}) (r = {:.2f} mm)\n'.format(len(selection_map),n_min,str(fit).replace('FitType.',''),event_map[k][l],k,l,r_pos)
#                 textstr_2 = 'e0 = {:.2f} \u00B1 {:.2f}\nLt = {:.2f} \u00B1 {:.2f}\nChi2 = {:.2f}'.format(myfloat(fMAP[k][l].e0),myfloat(fMAP[k][l].e0u),myfloat(fMAP[k][l].lt),myfloat(fMAP[k][l].ltu),myfloat(fMAP[k][l].c2))
#                 plt.text(320, 12200, textstr_1, fontsize=18);
#                 plt.text(780, 12300, textstr_2, fontsize=18);
                
# #                 plt.text(320, 11200, textstr_1, fontsize=18);
# #                 plt.text(780, 11300, textstr_2, fontsize=18);
# #                 plt.text(-20, 11200, textstr_1, fontsize=18);
# #                 plt.text(340, 11300, textstr_2, fontsize=18);
#                 plt.xlabel('Drift time ($\mu$s)',fontsize=18); plt.ylabel('S2e (pes)',fontsize=18)
#                 a = plt.hist2d(selection_map[k][l].Z,selection_map[k][l].S2e, 30, range=((0,500),range_e),cmap='coolwarm');
#                 plt.tick_params(axis='both',labelsize=18);
#                 cbar = plt.colorbar(a[3]);
#                 cbar.ax.tick_params(labelsize=18);
#                 cbar.set_label('Number of events',fontsize=18);
                
                
#                 plt.subplot(1,2,2)
#                 plt.hist(selection_map[k][l].S2e,40,range_e,label='Entries: {0}'.format(len(selection_map[k][l].S2e)), histtype = 'stepfilled');
#                 plt.xlabel('S2e (pes)',fontsize=18)
#                 plt.ylabel('Entries',fontsize=18)
#                 plt.axvline(x=np.mean(selection_map[k][l].S2e),color='r',label='Mean value = {:.2f}'.format(np.mean(selection_map[k][l].S2e)), linewidth = 2)
#                 plt.axvline(x=np.median(selection_map[k][l].S2e),color='g',label='Median value = {:.2f}'.format(np.median(selection_map[k][l].S2e)), linewidth = 2)
#                 plt.legend()
#                 plt.tick_params(axis='both',labelsize=18)

#                 plt.savefig('/home/afonso/data/results/7949_7950_7951_7952_7953_7954_7955_7956_results/fits_map_dst_corrected_50bins/{}_{}bins_{}xbin_{}ybin.png'.format(str(fit).replace('FitType.',''),len(fMAP),k,l), dpi=fig.dpi, bbox_inches = "tight")
                
# #                 plt.savefig('/home/afonso/data/results/7949_7950_7951_7952_7953_7954_7955_7956_results/fits_map_creation_50bins/{}_{}bins_{}xbin_{}ybin.png'.format(str(fit).replace('FitType.',''),len(fMAP),k,l), dpi=fig.dpi, bbox_inches = "tight")
# #                 plt.show();
                
                
#                 plt.close('all')
#             else:
#                 continue
    return fMAP


def fit_fcs_in_rphi_sectors_df(sector        : int,
                               selection_map : Dict[int, List[DataFrame]],
                               event_map     : DataFrame,
                               n_time_bins   : int,
                               time_diffs    : np.array,
                               nbins_z       : int,
                               nbins_e       : int,
                               range_z       : Tuple[float, float],
                               range_e       : Tuple[float, float],
                               energy        : str                 = 'S2e',
                               z             : str                 = 'Z',
                               fit           : FitType             = FitType.unbined,
                               n_min         : int                 = 100)->List[FitParTS]:
    """
    Returns fits to a (radial) sector of a RPHI-time series map

        Parameters
        ----------
            sector
                Radial sector where the fit is performed.
            selection_map
                A map of selected events defined as Dict[int, List[KrEvent]]
            event_map
                An event map defined as a DataFrame
            n_time_bins
                Number of time bins for the time series.
            time_diffs
                Vector of time differences for the time series.
            nbins_z
                Number of bins in Z for the fit.
            nbins_e
                Number of bins in energy.
            range_z
                Range in Z for fit.
            range_e
                Range in energy.
            energy:
                Takes two values: S2e (uses S2e field in kre) or E (used E field on kre).
                This field allows to select fits over uncorrected (S2e) or corrected (E) energies.
            fit
                Selects fit type.
            n_min
                Minimum number of events for fit.

        Returns
        -------
            A List[FitParTS], one FitParTs per PHI sector.

        @dataclass
        class FitParTS:             # Fit parameters Time Series
            ts   : np.array          # contains the time series (integers expressing time differences)
            e0   : np.array          # e0 fitted in time series
            lt   : np.array
            c2   : np.array
            e0u  : np.array          # e0 error fitted in time series
            ltu  : np.array

    """

    wedges    =[len(kre) for kre in selection_map.values() ]  # number of wedges per sector
    tfrst     = time_diffs[0]
    tlast     = time_diffs[-1]

    fps =[]
    for i in range(wedges[sector]):
        if event_map[sector][i] > n_min:
            ts, masks =  get_time_series_df(n_time_bins, (tfrst, tlast), selection_map[sector][i])
            fp  = time_fcs_df(ts, masks, selection_map[sector][i],
                              nbins_z, nbins_e, range_z, range_e, energy, z, fit)
        else:
            warnings.warn(f'Cannot fit: events in s/w[{sector}][{i}] ={event_map[sector][i]} < {n_min}',
                         UserWarning)

            dum = np.zeros(len(ts), dtype=float)
            dum.fill(np.nan)
            fp  = FitParTS(ts, dum, dum, dum, dum, dum)

        fps.append(fp)
    return fps
