from __future__ import division
from builtins import zip
from builtins import range
from pybasicbayes.abstractions import GibbsSampling
import numpy as np
from warnings import warn
from pypolyagamma import PyPolyaGamma
__all__ = ['Dynamic_GLM']

def local_multivariate_normal_draw(x, sigma, normal):
    """
    Function to combine pre-drawn Normals (normal) with the desired mean x and variance sigma
    Cholesky doesn't like 0 cov matrix, but we want it.

    This might need changing if in practice we see plently of 0 matrices
    """
    try:
        return x + np.linalg.cholesky(sigma).dot(normal)
    except np.linalg.LinAlgError:
        if np.isclose(sigma, 0).all():
            return x
        else:
            print("Weird covariance matrix")
            quit()


ppgsseed = 4
if ppgsseed == 4:
    print("Using default seed")
ppgs = PyPolyaGamma(ppgsseed)


class Dynamic_GLM(GibbsSampling):
    """
    This class enables a drifting input output iHMM with logistic link function.

    States are thus dynamic GLMs, giving us more freedom as to the inputs we give the model.

    Hyperparameters:

        n_regressors: number of regressors for the GLM
        T: number of timesteps (sessions)
        prior_mean: mean of regressors at the beginning (usually 0 vector)
        P_0: variance of regressors at the beginning
        Q: variance of regressors between timesteps (can be different across steps, but we use the same matrix throughout)
        jumplimit: for how many timesteps after last being used are the state weights allowed to change
    """

    def __init__(self, n_regressors, T, prior_mean, P_0, Q, jumplimit=1):

        self.n_regressors = n_regressors
        self.T = T
        self.jumplimit = jumplimit
        self.x_0 = prior_mean
        self.P_0, self.Q = P_0, Q
        self.psi_diff_saves = []  # this can be used to resample the variance, but is currently unused
        self.noise_mean = np.zeros(self.n_regressors)  # save this, so as to not keep creating it
        self.identity = np.eye(self.n_regressors)  # not really needed, but kinda useful for state sampling

        self.weights = np.empty((self.T, self.n_regressors))
        self.weights[0] = np.random.multivariate_normal(mean=self.x_0, cov=self.P_0)
        for t in range(1, T):
            self.weights[t] = self.weights[t - 1] + np.random.multivariate_normal(mean=self.noise_mean, cov=self.Q[t - 1])

    def rvs(self, inputs, times):
        """Given the input features and their time points, create responses from the dynamic GLM weights."""
        outputs = []
        for input, t in zip(inputs, times):
            if input.shape[0] == 0:
                output = np.zeros((0, 1))
            else:
                # find the distinct sets of features, how often they exist, and how to put the answers back in place
                types, inverses, counts = np.unique(input, return_inverse=True, return_counts=True, axis=0)

                # draw responses
                output = np.append(input, np.empty((input.shape[0], 1)), axis=1)
                for i, (type, c) in enumerate(zip(types, counts)):
                    temp = np.random.rand(c) < 1 / (1 + np.exp(- np.sum(self.weights[t] * type)))
                    output[inverses == i, -1] = temp
            outputs.append(output)
        return outputs

    def log_likelihood(self, input, timepoint):
        predictors, responses = input[:, :-1], input[:, -1]
        nans = np.isnan(responses)
        probs = np.zeros((input.shape[0], 2))
        out = np.zeros(input.shape[0])
        # I could possibly save the 1 / ..., since it's logged it's just - log (but the other half of the probs is an issue)
        probs[:, 1] = 1 / (1 + np.exp(- np.sum(self.weights[timepoint] * predictors, axis=1)))
        probs[:, 0] = 1 - probs[:, 1]
        # probably not necessary, just fill everything with probs and then have some be 1 - out?
        out[~nans] = probs[np.arange(input.shape[0])[~nans], responses[~nans].astype(int)]
        out = np.clip(out, np.spacing(1), 1 - np.spacing(1))
        out[nans] = 1

        return np.log(out)

    # Gibbs sampling
    def resample(self, data=[]):
        # TODO: Clean up this mess, I always have to call delete_obs_data because of all the saved shit!
        self.psi_diff_saves = []
        summary_statistics, all_times = self._get_statistics(data)
        types, pseudo_counts, counts = summary_statistics

        # if state is never active, resample from prior, but without dynamic change
        if len(counts) == 0:
            self.weights = np.tile(np.random.multivariate_normal(mean=self.x_0, cov=self.P_0), (self.T, 1))
            return

        """compute Kalman filter parameters, also sort out which weight vector goes to which timepoint"""
        timepoint_map = {}
        total_types = 0
        actual_obs_count = 0
        change_points = []
        prev_t = all_times[0] - 1
        fake_times = []
        for type, t in zip(types, all_times):
            if t > prev_t + 1:
                add_list = list(range(total_types, min(total_types + t - prev_t - 1, total_types + self.jumplimit)))
                change_points += add_list
                fake_times += add_list
                for i, sub_t in enumerate(range(total_types, total_types + t - prev_t - 1)): # TODO: set up this loop better
                    timepoint_map[prev_t + i + 1] = min(sub_t, total_types + self.jumplimit - 1)
                total_types += min(t - prev_t - 1, self.jumplimit)
            total_types += type.shape[0]
            actual_obs_count += type.shape[0]
            change_points.append(total_types - 1)
            timepoint_map[t] = total_types - 1
            prev_t = t

        self.pseudo_Q = np.zeros((total_types, self.n_regressors, self.n_regressors))
        # TODO: is it okay to cut off last timepoint here?
        for k in range(self.T):
            if k in timepoint_map:
                self.pseudo_Q[timepoint_map[k]] = self.Q[k]  # for every timepoint, map it's variance onto the pseudo_Q

        """sample pseudo obs"""
        temp = np.empty(actual_obs_count)
        psis = np.empty(actual_obs_count)
        psi_count = 0
        predictors = []
        for type, time in zip(types, all_times):
            for t in type:
                psis[psi_count] = np.sum(self.weights[time] * t)
                predictors.append(t)
                psi_count += 1

        ppgs.pgdrawv(np.concatenate(counts).astype(float), psis, temp)
        self.R = np.zeros(total_types)
        mask = np.ones(total_types, dtype=np.bool)
        mask[fake_times] = False
        self.R[mask] = 1 / temp
        self.pseudo_obs = np.zeros(total_types)
        self.pseudo_obs[mask] = np.concatenate(pseudo_counts) / temp
        self.pseudo_obs = self.pseudo_obs.reshape(total_types, 1)
        self.H = np.zeros((total_types, self.n_regressors, 1))
        self.H[mask] = np.array(predictors).reshape(actual_obs_count, self.n_regressors, 1)

        """compute means and sigmas by filtering"""
        # if there is no obs, sigma_k = sigma_k_k_minus and x_hat_k = x_hat_k_k_minus (because R is infinite at that time)
        self.compute_sigmas(total_types)
        self.compute_means(total_types)

        """sample states"""
        self.weights.fill(0)
        pseudo_weights = np.empty((total_types, self.n_regressors))
        pseudo_weights[total_types - 1] = np.random.multivariate_normal(self.x_hat_k[total_types - 1], self.sigma_k[total_types - 1])

        normals = np.random.standard_normal((total_types - 1, self.n_regressors))
        for k in range(total_types - 2, -1, -1):  # normally -1, but we already did first sampling
            if np.all(self.pseudo_Q[k] == 0):
                pseudo_weights[k] = pseudo_weights[k + 1]
            else:
                updated_x = self.x_hat_k[k].copy()  # not sure whether copy is necessary here
                updated_sigma = self.sigma_k[k].copy()

                for m in range(self.n_regressors):
                    epsilon = pseudo_weights[k + 1, m] - updated_x[m]
                    state_R = updated_sigma[m, m] + self.pseudo_Q[k, m, m]

                    updated_x += updated_sigma[:, m] * epsilon / state_R  # I don't think it's important, but probs we need the first column
                    updated_sigma -= updated_sigma.dot(np.outer(self.identity[m], self.identity[m])).dot(updated_sigma) / state_R

                pseudo_weights[k] = local_multivariate_normal_draw(updated_x, updated_sigma, normals[k])

        for k in range(self.T):
            if k in timepoint_map:
                self.weights[k] = pseudo_weights[timepoint_map[k]]

        """Sample before and after active times too"""
        for k in range(all_times[0] - 1, -1, -1):
            if k > all_times[0] - self.jumplimit - 1:
                self.weights[k] = self.weights[k + 1] + np.random.multivariate_normal(self.noise_mean, self.Q[k])
            else:
                self.weights[k] = self.weights[k + 1]
        for k in range(all_times[-1] + 1, self.T):
            if k < min(all_times[-1] + 1 + self.jumplimit, self.T):
                self.weights[k] = self.weights[k - 1] + np.random.multivariate_normal(self.noise_mean, self.Q[k])
            else:
                self.weights[k] = self.weights[k - 1]

        return pseudo_weights
        # If one wants to resample variance...
        # self.psi_diff_saves = np.concatenate(self.psi_diff_saves)

    def _get_statistics(self, data):
        # TODO: improve
        summary_statistics = [[], [], []]
        times = []
        if isinstance(data, np.ndarray):
            warn('What you are trying is probably stupid, at least the code is not implemented')
            quit()
        else:
            for i, d in enumerate(data):
                clean_d = d[~np.isnan(d[:, -1])]
                if len(clean_d) != 0:
                    predictors, responses = clean_d[:, :-1], clean_d[:, -1]
                    types, inverses, counts = np.unique(predictors, return_inverse=True, return_counts=True, axis=0)
                    pseudo_counts = np.zeros(len(types))
                    for j, c in enumerate(counts):
                        mask = inverses == j
                        pseudo_counts[j] = np.sum(responses[mask]) - c / 2
                    summary_statistics[0].append(types)
                    summary_statistics[1].append(pseudo_counts)
                    summary_statistics[2].append(counts)
                    times.append(i)

        return summary_statistics, times

    def compute_sigmas(self, T):
        """Sigmas can be precomputed (without z), we do this here."""
        # We rely on the fact that H.T.dot(sigma).dot(H) is just a number, no matrix inversion needed
        # furthermore we use the fact that many matrices are identities, namely F and G
        self.sigma_k = []  # we have to reset this for repeating this calculation later for the resampling (R changes)
        self.sigma_k_k_minus = [self.P_0]
        self.gain_save = []
        for k in range(T):
            if self.R[k] == 0:
                self.gain_save.append(None)
                self.sigma_k.append(self.sigma_k_k_minus[k])
                self.sigma_k_k_minus.append(self.sigma_k[k] + self.pseudo_Q[k])
            else:
                sigma, H = self.sigma_k_k_minus[k], self.H[k]  # we will need this a lot, so shorten it
                gain = sigma.dot(H).dot(1 / (H.T.dot(sigma).dot(H) + self.R[k]))
                self.gain_save.append(gain)
                self.sigma_k.append(sigma - gain.dot(H.T).dot(sigma))
                self.sigma_k_k_minus.append(self.sigma_k[k] + self.pseudo_Q[k])

    def compute_means(self, T):
        """Compute the means, the estimates of the states.
        Used to also contain self.x_hat_k_k_minus, but it's not necessary for our setup"""
        self.x_hat_k = [self.x_0]  # we have to reset this for repeating this calculation later for the resampling
        for k in range(T):  # this will leave out last state which doesn't have observation
            if self.gain_save[k] is None:
                self.x_hat_k.append(self.x_hat_k[k])
            else:
                x, H = self.x_hat_k[k], self.H[k]  # we will need this a lot, so shorten it
                self.x_hat_k.append(x + self.gain_save[k].dot(self.pseudo_obs[k] - H.T.dot(x)))

        self.x_hat_k.pop(0)  # remove initialisation element from list

    def num_parameters(self):
        return self.weights.size

    ### Max likelihood
    def max_likelihood(self, data, weights=None):
        warn('ML not implemented')


    def MAP(self, data, weights=None):
        warn('MAP not implemented')
