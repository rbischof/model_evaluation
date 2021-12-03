import numpy as np

from skill_metrics import centered_rms_dev
from skill_metrics import error_check_stats

def taylor_statistics(predicted,reference,field=''):
    '''
    Calculates the statistics needed to create a Taylor diagram as 
    described in Taylor (2001) using the data provided in the predicted 
    field (PREDICTED) and the reference field (REFERENCE).
    
    The statistics are returned in the STATS dictionary.

    If a dictionary is provided for PREDICTED or REFERENCE, then 
    the name of the field must be supplied in FIELD.
  
    The function currently supports dictionaries, lists, and np.ndarray,
    types for the PREDICTED and REFERENCE variables.
 
    Input:
    PREDICTED : predicted field
    REFERENCE : reference field
    FIELD     : name of field to use in PREDICTED and REFERENCE dictionaries
                (optional)
    NORM      : logical flag specifying statistics are to be normalized 
                with respect to standard deviation of reference field
                = True,  statistics are normalized
                = False, statistics are not normalized
 
    Output:
    STATS          : dictionary containing statistics
    STATS['ccoef'] : correlation coefficients (R)
    STATS['crmsd'] : centered root-mean-square (RMS) differences (E')
    STATS['sdev']  : standard deviations
 
    Each of these outputs are one-dimensional with the same length. 
    First index corresponds to the reference series for the diagram. 
    For example SDEV[1] is the standard deviation of the reference 
    series (sigma_r) and SDEV[2:N] are the standard deviations of the 
    other (predicted) series.
 
    Reference:
    
    Taylor, K. E. (2001), Summarizing multiple aspects of model 
      performance in a single diagram, J. Geophys. Res., 106(D7),
      7183-7192, doi:10.1029/2000JD900719.

    Created on Dec 3, 2016
    Revised on Dec 3, 2021
    
    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com
    Modified by: Rafael Bischof
        Institute of Structural Engineering (IBK)
        https://ibk.ethz.ch/
        rabischof@ethz.ch

    '''

    p, r = error_check_stats(predicted,reference,field)

    # Calculate correlation coefficient
    ccoef = np.corrcoef(p,r)
    ccoef = ccoef[0]

    # Calculate centered root-mean-square (RMS) difference (E')^2
    crmsd = [0.0, centered_rms_dev(p,r)]

    # Calculate standard deviation of predicted field w.r.t N (sigma_p)
    sdevp = np.std(p)
    
    # Calculate standard deviation of reference field w.r.t N (sigma_r)
    sdevr = np.std(r)
    sdev = [sdevr, sdevp];

    # Store statistics in a dictionary
    stats = {'ccoef': ccoef, 'crmsd': crmsd, 'sdev': sdev}
    return stats
