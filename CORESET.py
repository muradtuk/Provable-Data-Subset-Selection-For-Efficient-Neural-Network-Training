"""****************************************************************************************************
MIT License
Copyright (c) 2023 Murad Tukan, Samson Zhou, Alaa Maalouf, Daniela Rus, Vladimir Braverman, Dan Feldman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************************************"""
import numpy as np
import copy
import time
from scipy import optimize
from datetime import datetime
from RegressionCoresets import computeSensitivity
import AproxMVEE

R = 10


def obtainSensitivity(X, w, approxMVEE=False):
    if not approxMVEE:
        return computeSensitivity(X, w)
    else:
        cost_func = lambda x: np.linalg.norm(np.dot(X, x), ord=1)
        mvee = AproxMVEE.MVEEApprox(X, cost_func, 3)
        ellipsoid, center = mvee.compute_approximated_MVEE()
        U = X.dot(ellipsoid)
        return np.linalg.norm(U, ord=1, axis=1)


def generateCoreset(X, y, sensitivity, sample_size, weights=None, SEED=1):
    if weights is None:
        weights = np.ones((X.shape[0], 1)).flatten()

    # Compute the sum of sensitivities.
    t = np.sum(sensitivity)

    # The probability of a point prob(p_i) = s(p_i) / t
    probability = sensitivity.flatten() / t

    startTime = time.time()

    # initialize new seed
    np.random.seed()

    # Multinomial Distribution
    hist = np.random.choice(np.arange(probability.shape[0]), size=sample_size, replace=False, p=probability.flatten())
    indxs, counts = np.unique(hist, return_counts=True)
    S = X[indxs]
    labels = y[indxs]

    # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
    weights = np.asarray(np.multiply(weights[indxs], counts), dtype=float).flatten()

    weights = np.multiply(weights, 1.0 / (probability[indxs] * sample_size))
    timeTaken = time.time() - startTime

    return indxs, S, labels, weights, timeTaken
