## The components

[Generalized additive models](https://projecteuclid.org/euclid.ss/1177013604) were formalized
in their namesake paper by Hastie and Tibshirani in 1986. They replace
the terms in linear regression with smooth functions of the predictors.
The form of those functions determines the structure, and thus
flexibility, of the model. When applied to univariate time series, the
observed variable is deconstructed into smooth functions of time. There
are some functions that are common to many time series, which we will
outline here. Schematically, the model we are describing looks something
like this:

![A schematic equation describing a generalized additive model of a time series.](figures/structural-eqn.png)

Let's take a brief look at each of the components that contribute to the
observed time series values.

### Trend

Over a long enough window of time, many time series display an overall
upward or downward trend. Even when this is not the case (i.e., the time
series is globally flat), there are often "local" trends that are active
during only part of the time series.

Many possible functions could be used to model trend, so selecting an
appropriate function is up to the modeler. For instance, it may be
apparent from the data that the observed quantity is growing
exponentially---in which case, an exponential function of time would be
fitting.

When the trend of a time series is more complex than overall increase or
decline, it may be modeled in a piecewise fashion. It is possible to
model very nonlinear global tendencies with piecewise linear
approximations, but we must be careful not to overfit; after all, any
function may be approximated well with small enough linear segments. As
such, we should include some notion of how likely a changepoint (a point
where the trend changes) is. One means of doing so is to start with many
potential changepoints, but put a sparsity-inducing prior^[A sparsity-inducing
prior is a prior distribution for the probability of each changepoint that is
highly peaked at zero; a
[Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution)
is often used. With such a prior, if we consider the vector of potential
changepoints, it will likely turn out sparse (having many zeros).] over
their magnitude, so that only a subset is ultimately selected.


![A smooth time series may be modeled with a piecewise linear approximation. This is particularly useful for capturing changing trends.](/figures/ff16-04.png)

Many processes have an intrinsic limit in capacity, above (or below)
which it is impossible for them to grow (or fall). These saturating
forecasts can be modeled with a logistic function. For instance, a
service provider can serve only so many customers; even if their growth
looks linear or exponential to begin with, a fixed upper limit exists.
An advantage of a structural approach is that we can encode this kind of
domain knowledge into the components we use to model a time series. In
contrast, black-box learners like neural networks can encode no such
knowledge. The trade-off is that when using a structural model, we
*must* specify the structure, whereas a recurrent neural network can
learn arbitrary functions.

### Seasonality

![Seasonal patterns may follow the natural seasons, but more generally refer to any repeating pattern. For instance, in time series of commercial activity, there is often a weekly pattern, where the weekends behave differently than the weekdays.](/figures/ff16-05.png)

Structural approaches to time series are especially useful when the time
series displays some seasonal periodicity. These may correspond to the
natural seasons, but for the purpose of modeling, a seasonal effect is
anything that is periodic---a repeating pattern. In the natural world,
many things (for example, the tides) exhibit an annual, monthly, or
daily cycle, corresponding to changes caused by the relative motion of
the Sun, Earth, and Moon. Likewise, in the world of commerce and
business, many phenomena repeat weekly, while often demonstrating very
different behaviour on weekdays and weekends.

To encode arbitrary periodic patterns, we need a flexible periodic
function. Any periodic function can be approximated by Fourier series. A
Fourier series is a weighted sum of sine and cosine terms with
increasingly high frequencies. Including more terms in the series
increases the fidelity of the approximation. We can tune how flexible
the periodic function is by increasing the *degree* of the approximation
(increasing the number of sine and cosine terms included).

A Fourier expansion guarantees the periodicity of the component; the end
of a cycle transitions smoothly into the start of the next cycle. The
appropriate number of Fourier terms for a component may be set either by
intuition for how detailed a seasonal pattern is, or by brute
hyperparameter search for the best performance.

Generalized additive models may have multiple seasonal components, each
having its own periodicity. For instance, there may be a repeating
annual cycle, weekly cycle, and daily cycle, all active in the same time
series.

![The black square wave can be approximated with a Fourier expansion. The smooth green line is a low-degree approximation. The orange line, which follows the black square much more closely, is a high-degree approximation.](/figures/ff16-06.png)

### Impact effects

![A time series may have seemingly anomalous points that occur at particular times, such as on holidays, or which coincide with sporting events.](/figures/ff16-07.png)

Some time series have discrete impact effects, active only at specific
times. For instance, sales for some consumer products are likely to peak
strongly on Black Friday. This isn't part of a weekly recurring pattern;
sales don't peak to the same level on every Friday. The date of Black
Friday also moves annually. However, whenever it is Black Friday, sales
will spike.

We can model such discrete impact effects by including a constant term
for them in the model, but having that term only be active at the
appropriate time. Then, the coefficient of the term quantifies the
additional effect (positive *or* negative) of it being a certain day (or
hour, or other time period), having accounted for the seasonal and trend
components. This additional constant effect will be added any time the
indicator is active. This kind of component is especially useful for
modeling holidays, which occur every year, and often on a different day
of the week.

In order to learn such effects, we must have several examples of the
event or holiday. Otherwise, we'll introduce a new parameter for a
single data point, and the component will also fit any extra noise at
the time it is active.

### External regressors

Up to this point, we have been considering a strictly univariate time
series, where the only predictor of the observed variable is time.
However, treating time series modeling as curve fitting means that we
can include any extra (i.e., external) regressors we like, just as in
regular regression. In fact, the impact effects we just discussed are
really extra regressors that take binary values. When modeling
electricity demand, as we do below, we could include outdoor temperature
as an external regressor that is likely to carry a lot of information
(in our example, we do not do this, though we could almost certainly
improve our ultimate metrics by doing so).

There are a few things to bear in mind when adding regressors. The first
is the interpretability of the prediction. Each forecast value is the
sum of the components active at that point. Including extra regressors
makes interpretation of the model a little more complicated, since the
prediction no longer depends only on time, but also on the values of
extra regressors. Whether the more subtle interpretation is a worthwhile
trade-off for increased predictive power is a decision the modeler will
need to make, based on the problem they need to solve.

Second, including external regressors often introduces another
forecasting problem. If we would like to predict what the electricity
demand will be next week, and we rely on temperature as a predictor,
we'd better know what the temperature will be next week! Whereas some
predictors may be known ahead of time---the day of week for instance, or
whether the day is a national holiday---many predictors must themselves
be forecast. Naturally occurring examples (that relate to our
electricity demand forecast example) include temperature, humidity, or
wind speed, but any feature that we do not have reliable knowledge of
ahead of time engenders this problem.

This complication, however, is not insurmountable. We could create a
forecast depending on temperature, and then forecast multiple scenarios
corresponding to different predictions about the temperature. It would
be important, though, when producing a general forecast, to correctly
account for the additional uncertainty in the prediction that arises
from forecasting the features.
