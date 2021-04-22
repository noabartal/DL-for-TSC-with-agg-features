import CFSmethod.entropy_estimators as ee


def information_gain(f1, f2, input_type='dd'):
    """
    This function calculates the information gain, where ig(f1, f2) = H(f1) - H(f1\f2)

    :param f1: {numpy array}, shape (n_samples,)
    :param f2: {numpy array}, shape (n_samples,)
    :return: ig: {float}
    """
    if input_type == 'dd':
        ig = ee.entropyd(f1) - conditional_entropy(f1, f2, input_type='dd')

    if input_type == 'cd':
        ig = ee.entropy(f1) - conditional_entropy(f1, f2, input_type='cd')

    if input_type == 'cc':
        ig = ee.entropy(f1) - conditional_entropy(f1, f2,  input_type='cc')
    return ig


def conditional_entropy(f1, f2, input_type='dd'):
    """
    This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
    :param f1: {numpy array}, shape (n_samples,)
    :param f2: {numpy array}, shape (n_samples,)
    :return: ce {float} conditional entropy of f1 and f2
    """

    if input_type == 'dd':
        ce = ee.entropyd(f1) - ee.midd(f1, f2)

    if input_type == 'cd':
        ce = ee.entropy(f1) - ee.micd(f1, f2)

    if input_type == 'cc':
        ce = ee.entropy(f1) - ee.mi(f1, f2)

    return ce


def su_calculation(f1, f2, input_type='dd'):
    """
    This function calculates the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
    :param f1: {numpy array}, shape (n_samples,)
    :param f2: {numpy array}, shape (n_samples,)
    :return: su {float} su is the symmetrical uncertainty of f1 and f2
    """
    # calculate information gain of f1 and f2, t1 = ig(f1, f2)
    t1 = information_gain(f1, f2, input_type)

    if input_type == 'dd':
        # calculate entropy of f1
        t2 = ee.entropyd(f1)
        # calculate entropy of f2
        t3 = ee.entropyd(f2)
    if input_type == 'cd':
        # calculate entropy of f1
        t2 = ee.entropy(f1)
        # calculate entropy of f2
        t3 = ee.entropyd(f2)

    if input_type == 'cc':
        # calculate entropy of f1
        t2 = ee.entropy(f1)
        # calculate entropy of f2
        t3 = ee.entropy(f2)


    su = (2.0 * t1) / (t2 + t3)

    return su


