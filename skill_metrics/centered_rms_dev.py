import numpy as np

def centered_rms_dev(predicted,reference):
    '''
    Calculates the centered root-mean-square (RMS) difference between 
    two variables PREDICTED and REFERENCE (E'). The latter is calculated 
    using the formula:

    (E')^2 = sum_(n=1)^N [(p_n - mean(p))(r_n - mean(r))]^2/N

    where p is the predicted values, r is the reference values, and
    N is the total number of values in p & r. Note that p & r must
    have the same number of values.

    Input:
    PREDICTED : predicted field
    REFERENCE : reference field

    Output:
    CRMSDIFF : centered root-mean-square (RMS) difference (E')^2

    Created on Nov 24, 2016
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

    # Check that dimensions of predicted and reference fields match
    pdims= predicted.shape
    rdims= reference.shape
    if not np.array_equal(pdims,rdims):
        message = 'predicted and reference field dimensions do not' + \
            ' match.\n' + \
            'shape(predicted)= ' + str(pdims) + ', ' + \
            'shape(reference)= ' + str(rdims) + \
            '\npredicted type: ' + str(type(predicted))
        raise ValueError(message)

    # Calculate means
    pmean = np.mean(predicted)
    rmean = np.mean(reference)

    # Calculate (E')^2
    crmsd = np.square((predicted - pmean) - (reference - rmean))
    crmsd = np.sum(crmsd)/predicted.size
    crmsd = np.sqrt(crmsd)

    return crmsd
