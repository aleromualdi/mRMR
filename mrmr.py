import numpy as np
from operator import itemgetter

from directarray.featureselection.featureselectionbase import FeatureFilterBase
from directarray.stats.information_metrics import InformationMetrics


class MRMR(FeatureFilterBase):
    """
    This class extends the directarray FeatureFilterBase and implements the
    MRMR algorithm for feature-selection: http://ieeexplore.ieee.org/document/1453511/


    :param n_features: int (default=20)
        Number of feature to select. If None are provided, then all the
        features that are available are ranked/ordered.

    :param method: str {'MID', 'MIQ'} (fefault='MIQ')
        Two most used mRMR schemes: MID and MIQ represent the Mutual Information
        Difference and Quotient schemes, respectively, to combine the relevance
        and redundancy that are defined using mutual information.

    :param k_max: int (default=None)
        The maximum number of top-scoring features to consider.
        If None is provided, then all the features that are available are consider.


    Example:
    >>> from directarray.datastructures.dataset import DataSet
    >>> from directarray.featureselection.mrmr import MRMR
    >>> data_A = [[1, 1, 0, 1, 1, 1, 2, 1, 1, 1],
    ...           [4, 1, 0, 1, 1, 0, 1, 1, 1, 2],
    ...           [1, 2, 1, 2, 1, 1, 2, 2, 0, 2],
    ...           [4, 1, 0, 1, 3, 2, 0, 1, 1, 1],
    ...           [1, 2, 1, 2, 1, 1, 1, 2, 1, 0],
    ...           [3, 1, 1, 2, 0, 0, 0, 0, 0, 1],
    ...           [1, 0, 1, 2, 1, 1, 1, 1, 2, 1]]
    >>> data_B = [[0, 1, 11, 1, 2, 1, 2, 1, 1, 1],
    ...           [1, 1, 12, 1, 1, 0, 1, 1, 2, 1],
    ...           [2, 1, 11, 0, 5, 1, 1, 2, 1, 0],
    ...           [1, 2, 10, 1, 3, 2, 2, 1, 2, 1],
    ...           [2, 1, 11, 2, 1, 1, 1, 0, 1, 0],
    ...           [1, 2, 10, 2, 5, 1, 0, 2, 0, 1],
    ...           [1, 1, 12, 3, 1, 1, 1, 1, 1, 1]]
    >>> dset_A = DataSet.generate_from_array(np.asarray(data_A), ds_name="dataset A")
    >>> dset_B = DataSet.generate_from_array(np.asarray(data_B), ds_name="dataset B")

    >>> feature_filter = MRMR(n_features=2)
    >>> feature_filter.add_trainingdata(dset_A, "A")
    >>> feature_filter.add_trainingdata(dset_B, "B")
    >>> feature_filter.select()
    ['feature_3', 'feature_5']
    """

    def __init__(self, n_features=20, method='MID', k_max=None):

        super(MRMR, self).__init__()
        self.MRMR = None

        self.n_features = n_features

        if method not in ('MID', 'MIQ'):
            raise ValueError("method must be one of 'MID', MIQ'.")

        self.method = method
        self.k_max = k_max

    def _prepare_data(self, key_list=None):
        """
        Preprocessing function to convert DataSet to numpy array data structure.
        """

        datamatrix = []
        class_names = []

        sample_keys = None
        if key_list is None:
            sample_keys = self.training_set[0]['data'].get_keys()
        else:
            sample_keys = key_list

        for idx, data in enumerate(self.training_set):
            name = data["name"]
            samples = data["data"]

            for smp in samples:
                vec = []
                for k in sample_keys:
                    vec.append(smp[k])

                datamatrix.append(vec)
                class_names.append(name)

        np_data = np.array(datamatrix)
        y = np.searchsorted(np.sort(np.unique(class_names)), class_names)

        return np_data, y, sample_keys

    def _handle_selection(self, training_set, validation_set, *args, **kwargs):
        """

        :param training_set: DataSetVector|list, the training set
        :param validation_set: unused
        :param args: additional positional arguments
        :param kwargs: additional keyword arguments
        :return: list, A vector of indices sorted in descending order, where each index
                 represents the importance of the feature, as computed by the MRMR algorithm.

        :return:
        """
        self.training_set = training_set

        np_data, y, keys = self._prepare_data()
        num_dim = np_data.shape[1]

        k_max = min(num_dim, self.k_max)

        # calculate mutual mutual informaton between features and target
        MI_t = []
        for x in np_data.T:
            MI_t.append(InformationMetrics.mutual_information(x, y))

        MI_vals = sorted(enumerate(MI_t), key=itemgetter(1), reverse=True)

        # subset the data down to k_max
        sorted_mi_idxs = [i[0] for i in MI_vals]
        np_data_subset = np_data[:, sorted_mi_idxs[0:k_max]]

        # Max-Relevance first feature
        idx, MaxRel = MI_vals[0]

        MI_vars = {}
        MI_vars[idx] = []
        for x in np_data_subset.T:
            MI_vars[idx].append(InformationMetrics.mutual_information(x, np_data_subset[:, idx]))

        threshold = 0.8

        # find related values
        related = sorted(((i, v) for i, v in enumerate(MI_vars[idx]) if v > threshold and i != idx), key=itemgetter(1), reverse=True)

        mrmr_vals = [(idx, MaxRel, related)]

        mask_idxs = [idx]
        for k in range(min(self.n_features - 1, num_dim - 1)):
            idx, MaxRel, mrmr = max(((
                  idx, MaxRel,
                  np.nan_to_num(MaxRel - sum(MI_vars[j][idx] for j, _, _ in mrmr_vals) / len(mrmr_vals)) if self.method == 'MID' else
                  np.nan_to_num(MaxRel / sum(MI_vars[j][idx] for j, _, _ in mrmr_vals) / len(mrmr_vals)))
                for idx, MaxRel in MI_vals[1:] if idx not in mask_idxs), key=itemgetter(2))

            MI_vars[idx] = []
            for x in np_data_subset.T:
                MI_vars[idx].append(InformationMetrics.mutual_information(x, np_data_subset[:, idx]))

            # find related values
            related = sorted(((i, v) for i, v in enumerate(MI_vars[idx]) if v > threshold and i != idx), key=itemgetter(1), reverse=True)

            mrmr_vals.append((idx, mrmr, related))
            mask_idxs.append(idx)

        mrmr_vals_sorted = sorted(mrmr_vals, key=itemgetter(1), reverse=True)
        mrmr_vals_sorted_ = mrmr_vals_sorted[:self.n_features]

        sel_idx = [x[0] for x in mrmr_vals_sorted_]
        sel_keys = [keys[idx] for idx in sel_idx]

        return sel_keys
