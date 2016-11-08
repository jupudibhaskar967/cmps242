from __future__ import division
import numpy as np
from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin, check_grad
from scipy.optimize import fsolve, bisect
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from random import shuffle
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.optimize import newton_cg
from scipy.special import expit
from scipy.linalg import eigvalsh
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from scipy.linalg import solve_triangular
from sklearn.linear_model.logistic import ( _logistic_loss_and_grad, _logistic_loss,
                                            _logistic_grad_hess,)
import string
import math
import collections
from numpy import *

ROOT_DIR = "/home/karthik/PycharmProjects/cmps242/homeworks/hw3/data"
STOP_WORDS = set(stopwords.words("english"))
# TRAINING_DATASET = ["enron1", "enron2", "enron3", "enron4", "enron5"]
# SKIP_DATASET = ["enron1", "enron2", "enron3"]
SKIP_DATASET = []
TRAINING_DATASET = ["enron1"]
CLASSIFY_DATASET = "enron6"
FILE_COUNT = 1
PUNCTUATIONS = set(string.punctuation)
VOCAB = collections.OrderedDict()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@staticmethod
def union(*sets):
    union = collections.OrderedSet()
    union.union(*sets)
    return union

def union(self, *sets):
    for set in sets:
        self |= set

def remove_punctuation(word):
    return ''.join(ch for ch in word if ch not in PUNCTUATIONS)


def process_words(data, x_coord, y_coord, spam=False):
    word_count = 0
    word_dict = {}
    cleansed_data = cleanse(data)
    # word_tokens = re.split("\W+", processed_data)
    word_tokens = tokenize(cleansed_data)
    for word in word_tokens:
        if word not in STOP_WORDS and len(word) > 1 and not word.isdigit():
            word_dict[word] = word_dict.get(word, 0.0) + 1.0

    vector = []
    for word in VOCAB.keys():
        vector.append(word_dict.get(word, 0.0))

    x_coord = append(x_coord, vector, axis=0)
    if spam:
        # for spam, we mark the first column as 1
        y_coord = append(y_coord, [1, 0], axis=0)
    else:
        # for ham, we mark the second column as 1
        y_coord = append(y_coord, [0, 1], axis=0)
    return word_count, x_coord, y_coord


def tokenize(cleansed_data):
    return word_tokenize(unicode(cleansed_data, errors='ignore'))


def cleanse(data):
    processed_data = remove_punctuation(data)
    processed_data = processed_data.lower()
    return processed_data


def process_file(dir_path, file_end, x_coord, y_coord):
    document_count = 0
    total_word_count = 0
    for file_name in os.listdir(dir_path):
        if file_name.endswith(file_end):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as myfile:
                document_count += 1
                if FILE_COUNT > 0 and document_count > FILE_COUNT:
                    break
                data = "".join(line.rstrip() for line in myfile)
                # Skipping "Subject:" string at the beginning
                data = data[8:]
                word_count, x_coord, y_coord = process_words(data, x_coord, y_coord, file_end == 'spam.txt')
                total_word_count += word_count
    return document_count, total_word_count, x_coord, y_coord


class LogisticRegression():
    """ A simple logistic regression model with L2 regularization (zero-mean
    Gaussian priors on parameters). """

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
                 alpha=.1, synthetic=False):

        # Set L2 regularization strength
        self.alpha = alpha

        # Set the data.
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n = y_train.shape[0]

        # Initialization only matters if you don't call train().
        self.all_betas = []
        self.betas = np.random.randn(self.x_train.shape[1])
        # self.betas = np.random.randn(self.x_train.shape[0])

    def negative_lik(self, betas):
        return -1 * self.lik(betas)

    def lik(self, betas):
        """ Likelihood of the data under the current settings of parameters. """

        # Data likelihood
        l = 0
        for i in range(self.n):
            l += math.log(sigmoid(self.y_train[i] * \
                             np.dot(betas, self.x_train[i, :])))

        # Prior likelihood
        for k in range(0, self.x_train.shape[1]):
            l -= (self.alpha / 2.0) * self.betas[k] ** 2

        return l

    def lik_k(self, beta_k, k):
        """ The likelihood only in terms of beta_k. """

        new_betas = self.betas.copy()
        new_betas[k] = beta_k

        return self.lik(new_betas)

    def train(self):
        """ Define the gradient and hand it off to a scipy gradient-based
        optimizer. """

        # Define the derivative of the likelihood with respect to beta_k.
        # Need to multiply by -1 because we will be minimizing.
        dB_k = lambda B, k: (k > -1) * self.alpha * B[k] - np.sum([ \
                                                                      self.y_train[i] * self.x_train[i, k] * \
                                                                      sigmoid(-self.y_train[i] * \
                                                                              np.dot(B, self.x_train[i, :])) \
                                                                      for i in range(self.n)])

        # The full gradient is just an array of componentwise derivatives
        dB = lambda B: np.array([dB_k(B, k) \
                                 for k in range(self.x_train.shape[1])])

        # Optimize
        self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)

    def resample(self):
        """ Use slice sampling to pull a new draw for logistic regression
        parameters from the posterior distribution on beta. """

        failures = 0
        for i in range(10):
            try:
                new_betas = np.zeros(self.betas.shape[0])
                order = range(self.betas.shape[0])
                order.reverse()
                for k in order:
                    new_betas[k] = self.resample_beta_k(k)
            except:
                failures += 1
                continue

            for k in range(self.betas.shape[0]):
                self.betas[k] = new_betas[k]

            print self.betas,
            print self.lik(self.betas)

            self.all_betas.append(self.betas.copy())

        if failures > 0:
            "Warning: %s root-finding failures" % (failures)

    def resample_beta_k(self, k):
        """ Resample beta_k conditional upon all other settings of beta.
        This can be used in the inner loop of a Gibbs sampler to get a
        full posterior over betas.

        Uses slice sampling (Neal, 2001). """

        # print "Resampling %s" % k

        # Sample uniformly in (0, f(x0)), but do it in the log domain
        lik = lambda b_k: self.lik_k(b_k, k)
        x0 = self.betas[k]
        g_x0 = lik(x0)
        e = np.random.exponential()
        z = g_x0 - e

        # Find the slice of x where z < g(x0) (or where y < f(x0))
        # print "y=%s" % exp(z)
        lik_minus_z = lambda b_k: (self.lik_k(b_k, k) - z)

        # Find the zeros of lik_minus_k to give the interval defining the slice
        r0 = fsolve(lik_minus_z, x0)

        # Figure out which direction the other root is in
        eps = .001
        look_right = False
        if lik_minus_z(r0 + eps) > 0:
            look_right = True

        if look_right:
            r1 = bisect(lik_minus_z, r0 + eps, 1000)
        else:
            r1 = bisect(lik_minus_z, -1000, r0 - eps)

        L = min(r0, r1)
        R = max(r0, r1)
        x = (R - L) * np.random.random() + L

        # print "S in (%s, %s) -->" % (L, R),
        # print "%s" % x
        return x

    def training_reconstruction(self):
        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i, :]))

        return p_y1

    def test_predictions(self):
        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i, :]))

        return p_y1


class BayesianLogisticRegression(LinearClassifierMixin, BaseEstimator):
    '''
    Superclass for two different implementations of Bayesian Logistic Regression
    '''

    def __init__(self, n_iter, tol, fit_intercept, verbose):
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def fit(self, X, y):
        '''
        Fits Bayesian Logistic Regression
        Parameters
        -----------
        X: array-like of size (n_samples, n_features)
           Training data, matrix of explanatory variables

        y: array-like of size (n_samples, )
           Target values

        Returns
        -------
        self: object
           self
        '''
        # preprocess data
        X, y = check_X_y(X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # prepare for ovr if required
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = self._add_intercept(X)

        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_ = [0] * n_classes, [0] * n_classes
            self.intercept_ = [0] * n_classes
        else:
            self.coef_, self.sigma_, self.intercept_ = [0], [0], [0]

        # make classifier for each class (one-vs-the rest)
        for i in range(len(self.coef_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class = self.classes_[i]
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask] = self._mask_val
            coef_, sigma_ = self._fit(X, y_bin)
            if self.fit_intercept:
                self.intercept_[i], self.coef_[i] = self._get_intercept(coef_)
            else:
                self.coef_[i] = coef_
            self.sigma_[i] = sigma_

        self.coef_ = np.asarray(self.coef_)
        return self

    def predict_proba(self, X):
        '''
        Predicts probabilities of targets for test set

        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)

        Returns
        -------
        probs: numpy array of size [n_samples_test]
           Estimated probabilities of target classes
        '''
        # construct separating hyperplane
        scores = self.decision_function(X)
        if self.fit_intercept:
            X = self._add_intercept(X)

        # probit approximation to predictive distribution
        sigma = self._get_sigma(X)
        ks = 1. / (1. + np.pi * sigma / 8) ** 0.5
        probs = expit(scores.T * ks).T

        # handle several class cases
        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis=1), (probs.shape[0], 1))
        return probs

    def _add_intercept(self, X):
        '''Adds intercept to data matrix'''
        raise NotImplementedError

    def _get_intercept(self, coef):
        '''
        Extracts value of intercept from coefficients
        '''
        raise NotImplementedError

    def _get_sigma(self, X):
        '''
        Computes variance of predictive distribution (which is then used in
        probit approximation of sigmoid)
        '''
        raise NotImplementedError


class EBLogisticRegression(BayesianLogisticRegression):
    '''
    Implements Bayesian Logistic Regression with type II maximum likelihood
    (sometimes it is called Empirical Bayes), uses Gaussian (Laplace) method
    for approximation of evidence function.
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 50)
        Maximum number of iterations before termination

    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.

    solver: str, optional (DEFAULT = 'lbfgs_b')
        Optimization method that is used for finding parameters of posterior
        distribution ['lbfgs_b','newton_cg']

    n_iter_solver: int, optional (DEFAULT = 15)
        Maximum number of iterations before termination of solver

    tol_solver: float, optional (DEFAULT = 1e-3)
        Convergence threshold for solver (it is used in estimating posterior
        distribution),
    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations

    alpha: float (DEFAULT = 1e-6)
        Initial regularization parameter (precision of prior distribution)

    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    sigma_ : array, shape = (n_features, )
        eigenvalues of covariance matrix

    alpha_: float
        Precision parameter of weight distribution

    intercept_: array, shape = (n_features)
        intercept

    References:
    -----------
    [1] Pattern Recognition and Machine Learning, Bishop (2006) (pages 293 - 294)
    '''

    def __init__(self, n_iter=50, tol=1e-3, solver='lbfgs_b', n_iter_solver=15,
                 tol_solver=1e-3, fit_intercept=True, alpha=1e-6, verbose=False):
        super(EBLogisticRegression, self).__init__(n_iter, tol, fit_intercept, verbose)
        self.n_iter_solver = n_iter_solver
        self.tol_solver = tol_solver
        self.alpha = alpha
        if solver not in ['lbfgs_b', 'newton_cg']:
            raise ValueError(('Only "lbfgs_b" and "newton_cg" '
                              'solvers are implemented'))
        self.solver = solver
        # masking value (this is set for use in lbfgs_b and newton_cg)
        self._mask_val = -1.

    def _fit(self, X, y):
        '''
        Maximizes evidence function (type II maximum likelihood)
        '''
        # iterative evidence maximization
        alpha = self.alpha
        n_samples, n_features = X.shape
        w0 = np.zeros(n_features)

        for i in range(self.n_iter):

            alpha0 = alpha

            # find mean & covariance of Laplace approximation to posterior
            w, d = self._posterior(X, y, alpha, w0)
            mu_sq = np.sum(w ** 2)

            # use Iterative updates for Bayesian Logistic Regression
            # Note in Bayesian Logistic Gull-MacKay fixed point updates
            # and Expectation - Maximization algorithm give the same update
            # rule
            alpha = X.shape[1] / (mu_sq + np.sum(d))

            # check convergence
            delta_alpha = abs(alpha - alpha0)
            if delta_alpha < self.tol or i == self.n_iter - 1:
                break

        # after convergence we need to find updated MAP vector of parameters
        # and covariance matrix of Laplace approximation
        coef_, sigma_ = self._posterior(X, y, alpha, w)
        self.alpha_ = alpha
        return coef_, sigma_

    def _add_intercept(self, X):
        '''
        Adds intercept to data (intercept column is not used in lbfgs_b or newton_cg
        it is used only in Hessian)
        '''
        return np.hstack((X, np.ones([X.shape[0], 1])))

    def _get_intercept(self, coef):
        '''
        Returns intercept and coefficients
        '''
        return coef[-1], coef[:-1]

    def _get_sigma(self, X):
        ''' Compute variance of predictive distribution'''
        return np.asarray([np.sum(X ** 2 * s, axis=1) for s in self.sigma_])

    def _posterior(self, X, Y, alpha0, w0):
        '''
        Iteratively refitted least squares method using l_bfgs_b or newton_cg.
        Finds MAP estimates for weights and Hessian at convergence point
        '''
        n_samples, n_features = X.shape
        if self.solver == 'lbfgs_b':
            f = lambda w: _logistic_loss_and_grad(w, X[:, :-1], Y, alpha0)
            w = fmin_l_bfgs_b(f, x0=w0, pgtol=self.tol_solver,
                              maxiter=self.n_iter_solver)[0]
        elif self.solver == 'newton_cg':
            f = _logistic_loss
            grad = lambda w, *args: _logistic_loss_and_grad(w, *args)[1]
            hess = _logistic_grad_hess
            args = (X[:, :-1], Y, alpha0)
            w = newton_cg(hess, f, grad, w0, args=args,
                          maxiter=self.n_iter, tol=self.tol)[0]
        else:
            raise NotImplementedError('Liblinear solver is not yet implemented')

        # calculate negative of Hessian at w
        xw = np.dot(X, w)
        s = sigmoid(xw)
        R = s * (1 - s)
        Hess = np.dot(X.T * R, X)
        Alpha = np.ones(n_features) * alpha0
        if self.fit_intercept:
            Alpha[-1] = np.finfo(np.float16).eps
        np.fill_diagonal(Hess, np.diag(Hess) + Alpha)
        e = eigvalsh(Hess)
        return w, 1. / e


# ============== VB Logistic Regression (with Jaakola Jordan bound) ==================



def lam(eps):
    ''' Calculates lambda eps (used for Jaakola & Jordan local bound) '''
    return 0.5 / eps * (sigmoid(eps) - 0.5)


class VBLogisticRegression(BayesianLogisticRegression):
    '''
    Variational Bayesian Logistic Regression with local variational approximation.


    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 50 )
       Maximum number of iterations

    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold, if cange in coefficients is less than threshold
       algorithm is terminated

    fit_intercept: bool, optinal ( DEFAULT = True )
       If True uses bias term in model fitting

    a: float, optional (DEFAULT = 1e-6)
       Rate parameter for Gamma prior on precision parameter of coefficients

    b: float, optional (DEFAULT = 1e-6)
       Shape parameter for Gamma prior on precision parameter of coefficients

    verbose: bool, optional (DEFAULT = False)
       Verbose mode


    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    intercept_: array, shape = (n_features)
        intercepts

    References:
    -----------
   [1] Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
   [2] Murphy 2012, Machine Learning A Probabilistic Perspective ( Chapter 21 )
    '''

    def __init__(self, n_iter=50, tol=1e-3, fit_intercept=True,
                 a=1e-4, b=1e-4, verbose=True):
        super(VBLogisticRegression, self).__init__(n_iter, tol, fit_intercept, verbose)
        self.a = a
        self.b = b
        self._mask_val = 0.

    def _fit(self, X, y):
        '''
        Fits single classifier for each class (for OVR framework)
        '''
        eps = 1
        n_samples, n_features = X.shape
        XY = np.dot(X.T, (y - 0.5))
        w0 = np.zeros(n_features)

        # hyperparameters of q(alpha) (approximate distribution of precision
        # parameter of weights)
        a = self.a + 0.5 * n_features
        b = self.b

        for i in range(self.n_iter):
            # In the E-step we update approximation of
            # posterior distribution q(w,alpha) = q(w)*q(alpha)

            # --------- update q(w) ------------------
            l = lam(eps)
            w, Ri = self._posterior_dist(X, l, a, b, XY)

            # -------- update q(alpha) ---------------
            if self.fit_intercept:
                b = self.b + 0.5 * (np.sum(w[1:] ** 2) + np.sum(Ri[1:, :] ** 2))
            else:
                b = self.b + 0.5 * (np.sum(w ** 2) + np.sum(Ri ** 2))

            # -------- update eps  ------------
            # In the M-step we update parameter eps which controls
            # accuracy of local variational approximation to lower bound
            XMX = np.dot(X, w) ** 2
            XSX = np.sum(np.dot(X, Ri.T) ** 2, axis=1)
            eps = np.sqrt(XMX + XSX)

            # convergence
            if np.sum(abs(w - w0) > self.tol) == 0 or i == self.n_iter - 1:
                break
            w0 = w

        l = lam(eps)
        coef_, sigma_ = self._posterior_dist(X, l, a, b, XY, True)
        return coef_, sigma_

    def _add_intercept(self, X):
        '''Adds intercept to data matrix'''
        return np.hstack((np.ones([X.shape[0], 1]), X))

    def _get_intercept(self, coef):
        ''' Returns intercept and coefficients '''
        return coef[0], coef[1:]

    def _get_sigma(self, X):
        ''' Compute variance of predictive distribution'''
        return np.asarray([np.sum(np.dot(X, s) * X, axis=1) for s in self.sigma_])

    def _posterior_dist(self, X, l, a, b, XY, full_covar=False):
        '''
        Finds gaussian approximation to posterior of coefficients using
        local variational approximation of Jaakola & Jordan
        '''
        sigma_inv = 2 * np.dot(X.T * l, X)
        alpha_vec = np.ones(X.shape[1]) * float(a) / b
        if self.fit_intercept:
            alpha_vec[0] = np.finfo(np.float16).eps
        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        R = np.linalg.cholesky(sigma_inv)
        Z = solve_triangular(R, XY, lower=True)
        mean = solve_triangular(R.T, Z, lower=False)

        # is there any specific function in scipy that efficently inverts
        # low triangular matrix ????
        Ri = solve_triangular(R, np.eye(X.shape[1]), lower=True)
        if full_covar:
            sigma = np.dot(Ri.T, Ri)
            return mean, sigma
        else:
            return mean, Ri

def build_vocabulary(dir_path, file_end):
    document_count = 0
    total_word_count = 0
    for file_name in os.listdir(dir_path):
        if file_name.endswith(file_end):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as myfile:
                document_count += 1
                if FILE_COUNT > 0 and document_count > FILE_COUNT:
                    break
                data = "".join(line.rstrip() for line in myfile)
                # Skipping "Subject:" string at the beginning
                data = data[8:]

                cleansed_data = cleanse(data)
                # word_tokens = re.split("\W+", processed_data)
                word_tokens = tokenize(cleansed_data)
                for word in word_tokens:
                    if word not in STOP_WORDS and len(word) > 1 and not word.isdigit():
                        VOCAB[word] = 1.0
                        total_word_count += 1
    return document_count, total_word_count


def train_model(dataset_directory, x_coord, y_coord):
    spam_dir = os.path.join(dataset_directory, "spam")
    spam_doc_count, spam_word_count, x_coord, y_coord = process_file(spam_dir, "spam.txt", x_coord, y_coord)

    ham_dir = os.path.join(dataset_directory, "ham")
    ham_doc_count, ham_word_count,x_coord, y_coord = process_file(ham_dir, "ham.txt", x_coord, y_coord)
    return spam_doc_count, spam_word_count, ham_doc_count, ham_word_count, x_coord, y_coord


def add_to_vocabulary(dataset_dir):
    spam_dir = os.path.join(dataset_dir, "spam")
    build_vocabulary(spam_dir, "spam.txt")
    ham_dir = os.path.join(dataset_dir, "ham")
    build_vocabulary(ham_dir, "spam.txt")


def classify_spam(classify_dir):
    test_x_coord = np.zeros(len(VOCAB))
    test_y_coord = np.zeros(2)
    spam_dir = os.path.join(classify_dir, "spam")
    spam_document_count, spam_word_count,test_x_coord, test_y_coord = process_file(spam_dir, "spam.txt", test_x_coord, test_y_coord)
    return test_x_coord, test_y_coord, spam_document_count

def classify_ham(classify_dir):
    test_x_coord = np.zeros(len(VOCAB))
    test_y_coord = np.zeros(2)
    ham_dir = os.path.join(classify_dir, "ham")
    ham_document_count, ham_word_count,test_x_coord, test_y_coord = process_file(ham_dir, "ham.txt", test_x_coord, test_y_coord)
    return test_x_coord, test_y_coord, ham_document_count

if __name__ == '__main__':
    for filename in TRAINING_DATASET:
        training_dataset_dir = os.path.join(ROOT_DIR, filename)
        print("Adding words from dataset: %s to vocabulary" % training_dataset_dir)
        add_to_vocabulary(training_dataset_dir)

    spam_doc_count = 0
    spam_word_count = 0
    ham_doc_count = 0
    ham_word_count = 0

    x_coord = np.zeros(len(VOCAB))
    y_coord = np.zeros(2)
    for filename in TRAINING_DATASET:
        training_dataset_dir = os.path.join(ROOT_DIR, filename)
        print("Training model using dataset: %s" % training_dataset_dir)
        sd_count, sw_count, hd_count, hw_count,x_coord, y_coord = train_model(training_dataset_dir, x_coord, y_coord)
        spam_doc_count += sd_count
        spam_word_count += sw_count
        ham_word_count += hw_count
        ham_doc_count += hd_count



    print("spam docs: %d spam words: %d ham docs: %d ham words: %d" % (spam_doc_count, spam_word_count,
                                                                       ham_doc_count, ham_word_count))


    classify_dataset_dir = os.path.join(ROOT_DIR, CLASSIFY_DATASET)
    test_x_coord, test_y_coord, doc_count = classify_spam(classify_dataset_dir)
    # Create a new learner, but use the same data for each run
    lr = LogisticRegression(x_train=x_coord, y_train=y_coord,
                            x_test=test_x_coord, y_test=test_y_coord,
                            alpha=0.1)

    print "Initial likelihood:"
    print lr.lik(lr.betas)

    # Train the model
    lr.train()
    map_betas = lr.betas.copy()
    lr.plot_training_reconstruction()
    lr.plot_test_predictions()
    # model = BayesianLogisticRegression(5, 4, False, False)
    # model.fit(y_coord, x_coord)
    # result = model.predict(test_x_coord)

    split = array_split(test_y_coord, 2, axis=1)
    bincount(split)
    print("SPAM Error: %d out of total: %d" % (nonzero(test_y_coord), doc_count))