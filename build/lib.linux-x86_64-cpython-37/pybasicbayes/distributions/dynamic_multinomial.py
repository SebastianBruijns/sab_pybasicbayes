from __future__ import division
from builtins import zip
from builtins import range
__all__ = ['Dynamic_Input_Categorical']

from pybasicbayes.abstractions import \
    GibbsSampling

from scipy import sparse
import numpy as np
from warnings import warn
import time

from pgmult.lda import StickbreakingDynamicTopicsLDA
from pgmult.utils import psi_to_pi
from scipy.stats import invgamma


def enforce_limits(times, limit):
    times = np.array(times)
    diffs = np.zeros(len(times), dtype=np.int32)
    diffs[1:] = np.diff(times)
    diffs[diffs <= limit] = limit
    diffs -= limit
    diffs = np.cumsum(diffs)

    diffs += times[0]
    times -= diffs
    return times


assert np.array_equal(enforce_limits([0, 1, 2, 3, 4, 5], 3), [0, 1, 2, 3, 4, 5])
assert np.array_equal(enforce_limits([0, 2, 4, 6, 8, 10, 12, 14, 15], 3), [0, 2, 4, 6, 8, 10, 12, 14, 15])
assert np.array_equal(enforce_limits([0, 1, 2, 6, 10, 14], 3), [0, 1, 2, 5, 8, 11])
assert np.array_equal(enforce_limits([0, 1, 8, 20, 100], 3), [0, 1, 4, 7, 10])
assert np.array_equal(enforce_limits([0, 1, 8, 20, 100, 101, 102], 3), [0, 1, 4, 7, 10, 11, 12])
assert np.array_equal(enforce_limits([0, 1, 8, 20, 100, 101, 102, 104], 3), [0, 1, 4, 7, 10, 11, 12, 14])
assert np.array_equal(enforce_limits([0, 1, 8, 20, 100, 101, 102, 104, 110], 3), [0, 1, 4, 7, 10, 11, 12, 14, 17])
assert np.array_equal(enforce_limits([1, 8, 20, 100, 101, 102, 104, 110], 3), [0, 3, 6, 9, 10, 11, 13, 16])


def meh_time_info(all_times, sub_times, limit):
    # Return time stamps for the needed counts, considering jumps
    # Return list of mappings, to translate from this states timepoints to the overall timepoints
    # Return first and last timepoint
    times = []
    maps = []
    first, last = all_times[sub_times[0]], all_times[sub_times[-1]]
    jump_counter = 0
    time_counter = 0
    for i in range(first, last+1):
        if i in all_times:
            times.append(i)
            maps.append((time_counter, i))
            jump_counter = 0
            time_counter += 1
        else:
            jump_counter += 1
            if jump_counter <= limit:
                times.append(i)
                maps.append((time_counter, i))
                time_counter += 1
            else:
                maps.append((time_counter - 1, i))

    return times, maps, first, last


all_times, sub_times, limit = [0, 3, 9], [0, 2], 3
mya1, mya2, mya3, mya4 = [0, 1, 2, 3, 4, 5, 6, 9], [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (6, 7), (6, 8), (7, 9)], 0, 9
o1, o2, o3, o4 = meh_time_info(all_times, sub_times, limit)
assert mya1 == o1
assert mya2 == o2
assert mya3 == o3
assert mya4 == o4

all_times, sub_times, limit = [0, 3, 9], [1], 3
mya1, mya2, mya3, mya4 = [3], [(0, 3)], 3, 3
o1, o2, o3, o4 = meh_time_info(all_times, sub_times, limit)
assert mya1 == o1
assert mya2 == o2
assert mya3 == o3
assert mya4 == o4



def time_info(all_times, sub_times, limit):
    # Return list of mappings, to translate from this states timepoints to the overall timepoints
    # Return first and last timepoint
    maps = []
    first, last = all_times[sub_times[0]], all_times[sub_times[-1]]
    jump_counter = 0
    time_counter = 0
    for i in range(first, last+1):
        if i in all_times:
            maps.append((time_counter, i))
            jump_counter = 0
            time_counter += 1
        else:
            jump_counter += 1
            if jump_counter < limit:
                maps.append((time_counter, i))
                time_counter += 1
            else:
                maps.append((time_counter - 1, i))

    return maps, first, last


all_times, sub_times, limit = [0, 3, 9], [0, 2], 3
mya2, mya3, mya4 = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 9)], 0, 9
o2, o3, o4 = time_info(all_times, sub_times, limit)
assert mya1 == o1
assert mya2 == o2
assert mya3 == o3
assert mya4 == o4

all_times, sub_times, limit = [0, 3, 9], [1], 3
mya2, mya3, mya4 = [(0, 3)], 3, 3  # TODO: I don't think it matters whether first answer is [3] or [0]
o2, o3, o4 = time_info(all_times, sub_times, limit)
assert mya2 == o2
assert mya3 == o3
assert mya4 == o4
# TODO: more tests

class Dynamic_Input_Categorical(GibbsSampling):
    """
    This class enables a drifting input output iHMM.

    Everything is pretty much like in the input Categorical, but here we also
    allow drift across sessions for the Categoricals. Similar to a dynamic topic
    model (with one topic).

    We'll have to tricks quite a bit:
    - Instead of really resampling we'll reinstate a new dtm for every resampling and resample a couple of times (time killer)
    - We can't initialize from prior all that easily. We'll just instantiate a constant Dirichlet from prior
      once data is actually there, we'll use the real model
    - this will also be a bit hard to copy... maybe really just use package for resampling, otherwise have all that stuff saved in a list of Categoricals?
    Big problem: how to give log-likelihood for obs in sessions where this state was previously not present?
    We don't know what value this thing should have there, could be anything...
    -> if there is no data, we simply have to sample from prior. That is the will of Gibbs
    That is: sample from prior for first 3 unaccounted sessions, then leave constant
    But!: How to do this back in time? -> also Gibbs sample just from prior for three sessions

    ! First session cannot contain no data for StickbreakingDynamicTopicsLDA

    Idea: maybe save previously assigned betas (or psi's or whatever directly)
    then initialize new dtm from there, so as to have to do less resampling
    (this depends on what gets resampled first, hopefully the auxiliary variables first, then this saving should have an effect)

    Hyperparaemters:

        TODO

    Parameters:
        [weights, a vector encoding a finite, multidimensional pmf]
    """

    def __init__(self, n_inputs, n_outputs, T, sigmasq_states, jumplimit=3, n_resample=15):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs  # could be made different for different inputs
        if self.n_outputs != 2:
            warn('this requires adaptions in the row swapping code for the dynamic topic model')
            # quit()
        self.T = T
        self.jumplimit = jumplimit
        self.sigmasq_states = sigmasq_states
        self.n_resample = n_resample
        self.psi_diff_saves = []

        single_weights = np.zeros((self.n_inputs, self.n_outputs))
        for i in range(self.n_inputs):
            single_weights[i] = np.random.dirichlet(np.ones(self.n_outputs))
        self.weights = np.tile(single_weights, (self.T, 1, 1))  # init to constant over timepoints

    def rvs(self, input):
        print("Looks like simple copy from Input_Categorical, not useful, no dynamics")
        types, counts = np.unique(input, return_counts=True)
        output = np.zeros_like(input)
        for t, c in zip(types, counts):
            temp = np.random.choice(self.n_outputs, c, p=self.weights[t])
            output[input == t] = temp
        return np.array((input, output)).T

    def log_likelihood(self, x, timepoint):
        out = np.zeros(x.shape[0], dtype=np.double)
        nans = np.isnan(x[:, -1])
        err = np.seterr(divide='ignore')
        out[~nans] = np.log(self.weights[timepoint])[tuple(x[~nans].T.astype(int))]
        np.seterr(**err)
        out[nans] = 1
        return out

    # Gibbs sampling
    def resample(self, data=[]):
        self.psi_diff_saves = []
        counts, all_times = self._get_statistics(data)

        # if state is never active, resample from prior
        if counts.sum() == 0:
            single_weights = np.zeros((self.n_inputs, self.n_outputs))
            for i in range(self.n_inputs):
                single_weights[i] = np.random.dirichlet(np.ones(self.n_outputs))
            self.weights = np.tile(single_weights, (self.T, 1, 1))  # init to constant over timepoints
            return

        fake_times = enforce_limits(all_times, self.jumplimit)
        self.weights.fill(0)

        for i in range(self.n_inputs):
            if np.sum(counts[:, i]) == 0:
                self.weights[:, i] = np.random.dirichlet(np.ones(self.n_outputs))
            else:
                temp = np.sum(counts[:, i], axis=1)
                spec_times = np.where(temp)[0]
                maps, first_non0, last_non0 = time_info(all_times, spec_times, self.jumplimit)
                spec_fake_times = fake_times[spec_times]
                # we shuffle the columns around, so as to have the timeout answer first, for a hopefully more constistent variance estimation

                # dtm = StickbreakingDynamicTopicsLDA(sparse.csr_matrix(counts[spec_times, i][..., [2, 0, 1]]), spec_fake_times, K=1, alpha_theta=1, sigmasq_states=self.sigmasq_states)
                dtm = StickbreakingDynamicTopicsLDA(sparse.csr_matrix(counts[spec_times, i]), spec_fake_times, K=1, alpha_theta=1, sigmasq_states=self.sigmasq_states)

                for _ in range(self.n_resample):
                    dtm.resample()

                # save for resampling sigma
                self.psi_diff_saves.append(np.diff(dtm.psi, axis=0).ravel())

                # put dtm weights in right places
                for m in maps:
                    # shuffle back
                    # self.weights[m[1], i] = dtm.beta[m[0], :, 0][..., [1, 2, 0]]
                    self.weights[m[1], i] = dtm.beta[m[0], :, 0]

                sample = dtm.psi[0]
                for j in range(min(self.jumplimit, first_non0)):
                    sample += np.random.normal(0, np.sqrt(self.sigmasq_states), size=self.n_outputs - 1)[:, None]  # is this the correct way to do this? not sure
                    # self.weights[first_non0 - j - 1, i] = psi_to_pi(sample.T)[..., [1, 2, 0]]
                    self.weights[first_non0 - j - 1, i] = psi_to_pi(sample.T)
                if first_non0 > self.jumplimit:
                    # self.weights[:first_non0 - self.jumplimit, i] = psi_to_pi(sample.T)[..., [1, 2, 0]]
                    self.weights[:first_non0 - self.jumplimit, i] = psi_to_pi(sample.T)

                sample = dtm.psi[-1]
                for j in range(min(self.jumplimit, self.T - last_non0 - 1)):
                    sample += np.random.normal(0, np.sqrt(self.sigmasq_states), size=self.n_outputs - 1)[:, None]  # is this the correct way to do this? not sure
                    # self.weights[last_non0 + j + 1, i] = psi_to_pi(sample.T)[..., [1, 2, 0]]
                    self.weights[last_non0 + j + 1, i] = psi_to_pi(sample.T)
                if self.T - last_non0 - 1 > self.jumplimit:
                    # self.weights[last_non0 + self.jumplimit + 1:, i] = psi_to_pi(sample.T)[..., [1, 2, 0]]
                    self.weights[last_non0 + self.jumplimit + 1:, i] = psi_to_pi(sample.T)

        self.psi_diff_saves = np.concatenate(self.psi_diff_saves)
        assert np.count_nonzero(np.sum(self.weights, axis=2)) == np.sum(self.weights, axis=2).size

    def _get_statistics(self, data):
        # TODO: improve
        counts = []
        times = []
        timepoint_count = np.empty((self.n_inputs, self.n_outputs), dtype=int)
        if isinstance(data, np.ndarray):
            warn('What you are trying is probably stupid, at least the code is not implemented')
            quit()
            # assert len(data.shape) == 2
            # for d in data:
            #     counts[tuple(d)] += 1
        else:
            for i, d in enumerate(data):
                clean_d = (d[~np.isnan(d[:, -1])]).astype(int)
                if len(clean_d) != 0:
                    timepoint_count[:] = 0
                    for subd in clean_d:
                        timepoint_count[subd[0], subd[1]] += 1
                    counts.append(timepoint_count.copy())
                    times.append(i)

        return np.array(counts), times

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        warn('ML not implemented')

    def MAP(self,data,weights=None):
        warn('MAP not implemented')
