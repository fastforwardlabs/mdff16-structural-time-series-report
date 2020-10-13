## Evaluating time series models

We've briefly described the building blocks of generalized additive
models for time series. Before we fit a model to some data, we must
decide how to evaluate the fit. There are several considerations here
that are unique to time series data.

### Horizons

\[\[ figure: graph showing consequences of different evaluation points
\]\]

Time series forecasts are often evaluated at multiple time horizons. The
appropriate time horizon at which to evaluate a forecast depends
entirely on the ultimate purpose of the forecast. If we want to build a
model that will mostly be used to predict what will happen in the next
hour (with data in hourly intervals), we should evaluate the accuracy of
the forecast one step ahead. We could do this with a moving window
approach that first trains on an early part of the time series and
predicts a single step ahead. We would then evaluate those predictions
and move the training window one (or more) time steps on, and repeat the
process. This will allow us to build up a picture of how good a forecast
is at that particular time horizon (one step ahead).

Other times, we may be interested in how a forecast performs at multiple
horizons---in which case, we can simply repeat a version of the above,
but calculatie performance metrics multiple steps ahead each time.

For some time series problems, especially those on which we're likely to
adopt a curve fitting approach, *n*-step ahead forecasting does not make
sense, since the observations are unequally spaced, and there is no
notion of one time step. For long term forecasts, especially when
treated with a curve fitting approach, we may be interested not in
performance at some fixed horizon, but rather for a whole period---for
instance, the aggregate performance over a whole year (which is the
approach we'll take to electricity demand). In these cases, we may use
an evaluation process that looks more like what we do for regular
supervised learning model evaluation: separating train, test, and
validation data sets.

#### Do not cross-validate time series models

Data observed as a time series often carries a correlation through time;
indeed, this is what makes it worthwhile to treat as a time series,
rather than a collection of independent data points. When we perform
regular cross-validation, we leave out sections of the dataset, and
train on others. Doing so naively with time series, however, is very
likely to give us falsely confident results, because the split occurs on
the time variable; we may be effectively interpolating, or leaking trend
information backwards.^[Note that this applies equally well to any
train/validation/test data split of time series data.
We should test and validate on data that occurs after the training data.]
Time series data violates the assumption of independent and identically
distributed (i.i.d.) data that we often use implicitly in supervised learning.

For example, if we include both June and August in a fictitious training
set, but withhold July for testing, a good estimate of the model's true
performance is improbable, since the time series in July is likely to
interpolate between June and August. Essentially, by cutting out chunks
mid-way through the time series, we're testing a situation that will
never be the case when we come to actually predict, because we never
have information from the future. When we predict, we only have
information about the past, and we should evaluate our forecast
accordingly.

To perform the equivalent of cross-validation on a time series, we
should use a technique called *forward chaining*, or *rolling-origin*,
to evaluate a model trained on multiple segments of the data. To do so,
we first chunk our data in time, train on the first *n* chunks and
predict on the subsequent chunk, *n*+1. Then, we train on the first
*n*+1 chunks, and predict on chunk *n*+2, and so on. This prevents
patterns that are specific to some point in time from being leaked into
the test chunk for each train/predict step.

### Baselines

The process of developing a solution to almost any supervised learning
problem should begin with implementing a baseline model. The kind of
baseline to use depends on the problem. For instance, in a generic
classification problem, a sensible baseline is to always predict the
majority class. With that baseline model, we can compute all the
classification metrics we'd like. Then, any better model we build on top
of it can be compared against those metrics. A classification accuracy
of 91% may sound impressive, but if simply predicting the majority class
gives an accuracy of 95%, it's clear our model is junk!

An oft-used baseline for time series problems is a naive one-step ahead
forecast, where the prediction for each time step is simply the observed
value at the previous time step. This makes perfect sense when working
with short time scales, and when the thing we care about most is the
next step. However, in the kind of seasonal time series that an STS
model is good at dealing with, one-step ahead prediction is not so
useful; the model does not adapt on short time scales to new
information. Even if one were to re-fit the model on an additional data
point, it would be unlikely to change the forecast much, due to the
highly structured nature of the model. The strength of this kind of
structural model is in longer-term forecasting.

A sensible baseline is the seasonal naive forecast: rather than
predicting the value at each time step to be the same as the observed
value at the previous time step, we instead predict it as having the
same value as at the equivalent point in the previous period in the
season. For instance, if we have a daily periodicity, we could predict
that each hour in a day will have the same value as it did at the same
hour in the previous day. When dealing with multiple periods of
seasonality, to capture one full period of seasonality, we must offset
by the longest seasonality. For example, in the electricity demand
forecasting problem we tackle below, we use a baseline forecast of
predicting a repeated *year*, since there are strong patterns of
seasonality at the yearly, weekly, and daily level, and a year is the
longest of those seasons.

Once we have a sensible baseline for our time series, we can compute
metrics for the quality of a forecast. Many such metrics exist. Here, we
focus on two.

### Mean Absolute Percentage Error (MAPE)

The mean absolute percentage error is an intuitive measure of the
average degree to which a forecast is wrong. It has the advantage of
easy interpretation, with a MAPE of 0.05 corresponding to being about 5%
wrong, on average. The MAPE is defined exactly as we'd expect: it's the
mean of the absolute value of the error as a fraction of the true value.

\[\[ FIGURE: definition of MAPE. \]\]

![](figures/mape-defn.png)

MAPE, while interpretable, has some problems. It does not treat
overprediction and underprediction symmetrically. The same error (as in,
the same absolute difference between the forecast and true values) is
more heavily punished when it is an overprediction than an
underprediction. This can lead to selecting models that systematically
under-forecast. Further, error from over-forecasting is unbounded when
computing the MAPE: one can predict many times the true result, and MAPE
will punish that in proportion. Conversely, error from under-forecasting
is bounded; MAPE cannot allow more than a 100% under-forecast.

Further discussion of the shortcomings of MAPE (and other metrics) are
given in [Another look at measures of forecast accuracy](https://robjhyndman.com/papers/mase.pdf) by Hyndman and
Koehler, which also proposed a new measure of forecast accuracy: the
Mean Absolute Scaled Error.

### Mean Absolute Scaled Error (MASE)

The mean absolute scaled error can help overcome some of the
shortcomings of other measures of forecast quality. In particular, it
bakes in the naive one-step ahead baseline forecast by scaling the error
to the baseline error (the error of the baseline forecast) on the
training set. As a result, any forecast can easily be compared to the
baseline. If a method has a MASE greater than one, it is performing
worse than the baseline did on the training set. A MASE of less than one
is performing better. For instance, a MASE of 0.5 means that the method
has a mean error that is half the error of the baseline on the training
set.

MASE mitigates many of the problems with MAPE, including only going to
infinity (which MAPE does when the observed value is zero), when every
historical observation is exactly equal. This makes it particularly
appropriate for intermittent demand (see [Another look at forecast-accuracy metrics for intermittent demand](https://robjhyndman.com/papers/foresight.pdf)).
For more in-depth discussion of the considerations in choosing a
forecast-accuracy metric, we recommend the linked papers.

\[\[ FIGURE: definition of MASE
\]\]![](figures/mase-defn.png)

The baseline used when computing MASE is the naive one-step ahead
forecast, but it can easily be extended to a seasonal variant.