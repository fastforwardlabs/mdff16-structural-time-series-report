## Modeling electricity demand in California

To demonstrate the techniques outlined in our previous discussion, let's
tackle a real time series modeling problem: predicting electricity
demand in California. The [United States Energy Information Administration](https://www.eia.gov/)
provides detailed and open information about the energy use of each US state.
One can imagine myriad uses for accurate and timely projections of future energy
usage: from upscaling and downscaling production to discovering outlandishly
high or low usage in recent data.

![TODO: illustration of electricity use, maybe the inside of a
power plant with the time series graph on a control panel screen](/figures/ff16-09.png)

To create our forecast, we'll employ Facebook's open source forecasting
tool: [Prophet](https://facebook.github.io/prophet/).^[We first wrote about
Prophet around the time of its release in 2017:
[Taking Prophet for a Spin](https://blog.fastforwardlabs.com/2017/03/22/taking-prophet-for-a-spin.html).]

Prophet implements the components we described above: a generalized
additive model with piecewise linear trend, multiple layers of
seasonality modeled with Fourier series, and holiday effects. Under the
hood, the model is implemented in the probabilistic programming language
[Stan](https://mc-stan.org/). By default, it is not fully Bayesian,
using only a (maximum a posteriori) point estimate of parameters to
facilitate fast fitting. However, it does provide a measure of forecast
uncertainty, arising from two sources.

The first is the intrinsic noise in the observations, which is treated
as i.i.d., and normally distributed. In particular, Prophet makes no
attempt to model the error beyond that. It is possible to include
autoregressive terms in generalized additive models, but they come with
trade-offs (which we'll discuss later). The second is a novel approach
to uncertainty in the trend. When forecasting the future trend, the
model allows for changepoints with the same average frequency and
magnitude as was observed in the historic data. If uncertainty in the
components themselves is deemed desirable, the model can be fit with
MCMC to return a full posterior for model parameters.

For a deeper dive into how Prophet works, we recommend its
[documentation](https://facebook.github.io/prophet/), a paper called
[Forecasting at Scale](https://peerj.com/preprints/3190/), and
[this excellent twitter thread](https://twitter.com/seanjtaylor/status/1123278380369973248)
from Sean J. Taylor (one of the original authors of the paper).

### Forecasting in action: developing a model for electricity demand

Electricity demand forecasting has a whole body of academic work
associated with it.^[Recognising that probabilistic methods had historically
been underused in energy forecasting, the International Journal of
Forecasting issued a
[special request](http://blog.drhongtao.com/2013/10/probabilistic-energy-forecasting.html)
for probabilistic energy forecasts in 2013.]
In our case, the data serves as a real world
example of a demand time series, with clear periodic components that are
amenable to modeling with a generalized additive model. We'll use this
data as a test bed for where such models succeed and fail. Our intent is
to demonstrate the application of generalized additive models to a
forecasting problem and to describe the product implications for
probabilistic forecasts, rather than ship an actual product. When
tackling the same problem for the real world, we would advise consulting
the detailed electricity demand forecasting literature in much greater
depth than this report covers.

We'll expose the model diagnostics for each of the models we build as an
interactive dashboard, with top-line model comparison and the option to
deep dive into each model. This custom app sits in an adjacent space to
a pure experiment tracking system like MLFlow, and exhibits a philosophy
we heartily endorse at Cloudera Fast Forward: build tools for your work.
When you are not performing an ad-hoc bit of data science, but rather
building towards a potential product, long-lasting system, or even a
repeated report, it can pay dividends to think with the mindset of a
product developer. "What parts of the work I am doing are repeatable? If
I want a particular chart for this model, will I want it for all
models?" Building abstractions always takes time, but it is front-loaded
time. Once the model diagnostic suite is in place, we get model
evaluation and comparison for free for all later models. Assuming
modeling work will continue, this is likely a worthy investment of our
time. It also, conveniently, provides us with the ability to provide
screenshots as figures to support our discussion of the models, below.

#### The first approach: setting a baseline

As we mentioned earlier, before employing any sophisticated forecasting
model, it's important to always establish a reasonable baseline against
which to measure progress. For this application, we fit models on all
the data through 2018 and hold out the whole year of 2019 for testing.
Since we compare models on that same 2019 data set, we use 2020 YTD data
for performance reporting of the final model.

The demand time series has clear yearly, weekly, and daily seasonality
to be accounted for in our baseline. As such, our baseline forecast is
to assign for each hour of 2019 a demand corresponding to that which
occurs in the equivalent hour of 2018. Defining "equivalent" here is a
little tricky, since the day of the year does not align with the dates,
and there are strong weekday effects. When we settle for predicting
exactly 52 weeks ahead, this ends up performing reasonably
well---achieving a 8.31% MAPE on the holdout set.

Since this is effectively the same as the seasonal naive forecast, the
MASE on the training set is exactly one, by definition. This is the
baseline against which we scale other errors. The MASE of the baseline
on the holdout set is 1.07, indicating that the baseline performs worse
for 2019 than it does on average in the previous years.

\[\[ FIGURE: default prophet model against actuals \]\]

![](figures/baseline.png)

#### Applying Prophet

Our next model uses Prophet to capture trend and seasonality at three
levels: yearly, weekly, and daily. This is the default Prophet model,
which is designed to apply to a large number of forecasting problems,
and is often very good out of the box. Since we have several years of
hourly data, we expect the default priors over the scale of seasonal
variations to have little impact.

Again highlighting the importance of setting a baseline, this model
actually underperforms the baseline! It's MAPE is 8.85% on the holdout
set, with a MASE 1.12. Let's inspect the resulting forecast to
investigate why.

##### Debugging the fit

One of the benefits of treating a time series as a curve fitting
exercise is "debuggability." The top line metrics (MAPE and MASE) do not
really help here. They are important for model selection, but too coarse
to help us improve the fit. When the fit is not good, there are several
tools we can use to give us clues for how to improve it by iteratively
increasing complexity.

First, let's *look at the forecast* against some actuals in the hold out
test set. This can be incredibly revealing.^[At one point in the development of
our analysis, we had a programming error that messed up seasonality: we were
trying to fit a long-term periodicity to short-term variations. This was not
evident from top-line metrics; it simply looked to be a poorly-performing model.
One look at the forecast against actuals, however, reveals that the model was
not capturing within-day variations at all, and we had misspecified the duration
of a seasonal component.]

\[\[ FIGURE: screenshot of forecast vs test data \]\]

![](figures/prophet-simple.png)

There's an obvious spiky dip in the observed time series (depicted in
orange in the figure above) in late 2019. This is likely caused by a
power outage. There are several similar spikes of lower magnitude within
the series. In principle, we could manually remove those data points
entirely, and---since GAMs handle unevenly spaced observations
naturally---our model would still work. In practice, the highly
structured nature of the model makes it relatively robust to a handful
of outliers, so we won't remove them. (A practitioner with specialized
domain knowledge of electricity forecasting would likely know how best
to handle outliers.)

It is also useful to view a scatter plot of the forecast and predicted
values (effectively marginalizing out time). This chart should show us
if our model is preferentially overpredicting or underpredicting, and
how dispersed its predictions are. We will likely have to deal with some
overplotting issues, which we can do by either making the points
transparent enough to effectively result in a density plot, or by making
a density plot explicitly. The ideal predictor would produce a perfectly
straight diagonal line of points. Due to noise in the observations, no
model will provide a perfect fit; as such, the realistic goal is for the
line constructed of scattered points to be as narrow and straight as
possible. Here, the curve towards the *x*-axis indicates that we are
preferentially underpredicting.

\[\[ FIGURE: screenshot of scatter plot
\]\]![](figures/scatter.png)

We can also look at a histogram of the residuals (the difference between
forecast and prediction) to discover if we are favourably overplotting
or underplotting. This should be an approximately normal distribution,
and a better predictor will be strongly centred on zero and very narrow.
Any secondary peaks (or strong skew) will flag that we have missed some
systematic effect that may be causing us to consistently overpredict or
underpredict in some circumstances.

\[\[ FIGURE: screenshot of histogram \]\]

![](figures/histogram.png)

Finally, we can look at autocorrelation plots. The autocorrelation of a
time series tells us how correlated various time steps are with previous
time steps. We should look at the autocorrelation of the residuals (we'd
certainly expect there to be plenty of autocorrelation in the original
time series and forecast). If there are strong correlations in the
residuals, then we are effectively leaving signal unused.

\[\[ FIGURE: screenshot of autocorrelation plots \]\]

![](figures/autocorrelation.png)

![](figures/partial-autocorrelation.png)

When using an autoregressive model, highly autocorrelated residuals are
a sign of poor model fit. The same is true in the case of GAM-like
structural time series (without autoregressive terms), but it is more
expected. Autoregressive time series methods explicitly perform
regression from earlier time points to later ones. In contrast, the GAM
approach treats the points as independent and identically distributed,
and bakes the time series nature into the structure of the model
components. As such, with only a univariate time series input, we might
expect that some local correlations (for example, from one hour to the
next) are not caught. Nonetheless, autocorrelation can tell us which
patterns we are failing to fit. It simply may be the case that we cannot
fit those patterns with a structural time series that does not include
autocorrelation. In our case, we clearly have strong autocorrelation
within each day, and if we want to gain accuracy for forecasting the
next few hours, we should certainly include autoregressive terms.

::: info

#### So why not autoregress?

It could be done, even with a Prophet-like model.
In fact, we could implement it using the external
regressor functionality of Prophet by feeding in lagged observations.
However, in exchange for increased power to fit complex time series, we
give up several things. The first is interpretability. A given
prediction in Prophet is a sum of components that depend only on the
time. With an autoregressive component (or indeed, any external
regressor), the interpretation changes from "we predict *x* because it
is 4pm on a Wednesday in summer" to "we predict *x* because it is 4pm on
a Wednesday in summer, and the previous *n* points had some specific
values."

The second thing we sacrifice when we employ autoregression is the
ability to gracefully handle missing data. Since autoregression needs to
know the previous values of the time series, we must impute them through
some other means before they can be used. (In contrast, it is a benefit
of the curve fitting approach that we can naturally handle missing or
even unequally spaced data with no modifications or imputation needed.
After fitting, we can use the same forecasting method we use to predict
the future to impute any missing data, if we like.)

Finally, autoregression restricts our ability to forecast far into the
future. In order to regress on values, we must have those values. Using
one-step ahead forecasting repeatedly is likely to compound forecasting
errors very quickly, since we're regressing on our own predictions.

:::

#### Increasing complexity

By inspection of the forecast against past actuals, even on the training
set, we can see where the default Prophet model failed. It does not
capture the increased within-day variance in the summer months, and
exhibits too much variance in the winter months (see Figure X, where
actuals are illustrated in orange). We can improve this by changing the
structure of the model.

Inspection of the historical data shows that the variance increases a
lot within each day during summer, relative to winter. We allow for
different daily patterns in summer and winter by including separate
periodic components for each, and using an indicator variable to turn
the components on and off. This allows for four separate kinds of
within-day periodicities: weekdays in summer, weekends in summer,
weekdays in winter, and weekends in winter.

A downside of the model being so structured is that we need to specify
precisely what is meant by summer and winter. We use reasonable dates
corresponding to changes in the historical time series, defining the
summer patterns to apply from June 1st to September 30th. The true
transition is not so stark as that, so we expect our forecast to be less
reliable around those boundaries. Nonetheless, extending the model in
this way results in a better fit than both the default Prophet model and
the baseline, with a MAPE of 7.33% and a MASE of 0.94.

\[\[ FIGURE: improved model vs actuals \]\]

![](figures/prophet-complex.png)

#### The final model

Naturally, we won't stop exploring as soon as we beat the baseline. We
still have the problem of a slightly unsatisfactory effect, where the
overall variance of the forecast makes a step change between summer and
winter. Looking closely at the training data, we notice that the
within-day variance is proportional to the overall magnitude of the
forecast. When electricity demand was high in summer, the within-day
variation in demand was also high. When demand was lower in winter, the
within-day variation was also lower. This motivates us to use a
multiplicative interaction between trend and seasonal terms, rather than
an additive one.

While Prophet supports multiplicative interactions between the overall
trend and each seasonal term, it does not support multiplicative
interactions between different seasonal terms. Since our data has a
relatively flat trend, and the overall magnitude is determined
principally by the annual seasonality, we need to seek another means of
modeling this interaction.

Math comes to the rescue! We turn the fully additive model, where all
components add together, into a fully multiplicative model, where all
components combine multiplicatively (not just each component with the
trend) by modeling the *log* of the electricity demand. Then, to get
back to the true demand, we reverse the log transform by exponentiating
the predictions of the model. This turns additive terms for the log
demand into multiplicative terms for the true demand. When we apply this
transform to our dataset, we see a much smoother transition between
seasons, rather than the clear summer/winter split of the additive
model. It also improves the MAPE and MASE over all previous models, with
a MAPE of 6.95% and a MASE of 0.89.

\[\[ FIGURE: final model vs actuals \]\]

![](figures/prophet-complex-log)

Clearly the model is not perfect. However, with the diagnostic app, we
can investigate the areas of good and poor fit.

There seem to be two primary things to improve. One source of error is
that the predictions within a single day are the wrong shape; they are
either not bimodal enough or too bimodal. This seems to persist for
around a week at a time. The other source of error is the overall level
of the prediction being incorrect for a given week. Both of these
sources of error could likely be reduced with additional complexity in
the model. Perhaps more granular patterns than a summer/winter split
would work; It's very likely that including outdoor temperature as an
external regressor would improve the fit (this is a well-used feature in
the electricity demand forecasting literature), provided we could
reliably forecast the temperature.

One thing we note is that the in-sample error is lower than
out-of-sample. The MAPE for a full year in-sample (2018) is 4.40%,
whereas out-of-sample (2019), it is 6.95%. If the train and test data
are similarly distributed, this could be an indication of overfitting.
However, time series break the i.i.d. assumption; as such, we
hypothesize that it is more likely that 2019 is distributed differently
than the training data. One clue to this is that the out-of-sample
performance of all the models we tried is worse than the in-sample. It
may simply be, for instance, that there was a significant global trend
in 2019, whereas our prediction must average over many possible trends.

Having selected this model, we can report metrics on a held out slice of
the data that we did not use for model comparison. With a time series
model, we usually want to train with the most up-todate data. This means
that, unlike in the i.i.d. setting (where we can simply deploy the model
we have tested), we have the unfortunate circumstance of never being
able to fully assess the performance of the model we actually use. We
must trust the performance of the same model trained on less recent data
as a proxy. After retraining on all data up to 2020, we get a MAPE of
7.58% for 2020 (to October, the time of writing). The MASE, which is now
calculated from the new training set, and (as such) cannot be compared
to previous values, is 1.04, meaning this model performs worse on live
data than the benchmark did on training data. A more relevant comparison
is to the naive baseline on the testing data, which obtains a MAPE of
8.60%.

### Forecasting

With a sufficient model in hand (now trained on all the data we have
available), and its limitations understood, we can make forecasts.
Forecasting is intrinsically hard,^[The old Danish proverb applies here:
"It is difficult to make predictions, especially about the future."]
and it is wise to treat the
inherent uncertainty with the respect it deserves. Since the Prophet
model is accompanied by uncertainty bounds derived from the uncertainty
in trend and the noise, we should incorporate this in our forecast. As
such, exposing a simple point prediction would be a poor user interface
for our forecasts.^[We did experiment briefly with the full MCMC fit for uncertainty
bounds on all components. This is substantially more computationally
intensive, and took several hours to run. In this case, the large
amount of data available means that the posterior uncertainty on the
components is small, so we reverted back to the simpler, faster,
default MAP estimation.]

Instead, let us envision how this forecast might be used. Our imaginary
consumer is a person or team responsible for meeting the energy demands
of California. They would like to know the forecast demand, of course,
but also ask some more sophisticated questions. For instance, they would
like to be alerted if there is a greater than 5% chance that the demand
in any given hour of the week exceeds a specific amount. Point
predictions alone are insufficient to calculate and manage risk.

We can represent our forecast, and the associated uncertainty, by
sampling possible futures from the model. Note that we do not sample
each time point independently. Each sample is a full, coherent future,
for which we may compute whatever statistics we like. Working with
samples affords us the ability to answer precisely the sort of
probabilistic questions that an analyst working with the demand forecast
might ask.

For instance, to provide an alert for probable high energy usage in an
hour, we can simply draw 1000 samples, and filter these possible futures
to only those for which there is an hour that exceeds some threshold
that we choose. Then, we count the number of futures in which this is
true, and divide it by 1000 (the original number of futures we sampled),
and this gives us the probability of exceeding the threshold during at
least one hour in a given week. Note that anything computed based on
possible futures is subject to the assumptions of the model---and if the
model is misspecified (in practice, all models are; it's simply a
question of by how much), we should distrust the computations we make in
proportion to the degree of misspecification.

Assuming the model performs well-enough for our purposes, computing on
possible futures is extremely powerful. A point-like forecast answers
only one question: "What is the forecast at time *t*?" (or possibly an
aggregate of that, e.g., "What is the forecast for the next week?"). In
contrast, by sampling possible futures from the model, we can answer
queries like "How likely is it that we will require more than X
Megawatt-hours of energy next week?" alongside the point-like questions
of "What is the most likely prediction for time point *t*?" and "How
confident are we about the prediction?"

We present our electricity demand forecast with a simple app,
constructed, like our diagnostic app, using the open source tools
[Streamlit](https://www.streamlit.io/) and
[Plotly](https://plotly.com/). The app reads one thousand sample
forecasts (generated in batch offline) for one year ahead of the most
recent observation. It displays a zoomable chart of the mean of those
samples, and a selection of samples themselves to indicate the
associated uncertainty. We report the aggregate demand in any selected
time period (the sum of the hourly demand). We also expose a simple
probabilistic question that would be impossible to answer with a point
forecast: "What is the probability of the aggregate demand in the
selected range exceeding a chosen threshold?" We envision that extended
versions of this interface could provide a useful interface for analysts
using forecasting to aid, for example, capacity planning.

![TODO: app interface](figures/ff16-10.png)

#### Backcasting

Since our model depends only on time, we can run forecasts arbitrarily
far into the future (though we will increase our uncertainty in the
trend). Alternatively, we may "backcast" onto time periods that have
already happened. This is particularly useful for two applications:
imputing missing data, and anomaly detection.

##### Imputation

Situations in which data is missing can arise in a multitude of ways,
including that the data was never collected. Imagine, for instance, if
the telemetry for electricity demand simply failed for a few hours;
there is no way to recover base data that does not exist.

Some methods require that there be no missing data, and we often end up
imputing the missing data based on some average: perhaps the median
value for similar examples, where "similar" is defined by an ad-hoc
heuristic. For example, if we were missing a day of data, we might
impute each hour as the median for that day-of-week among the
surrounding two months. This heuristic is itself a model of the data,
and probably a less sophisticated one than the model we aim to construct
(after all, if it is more sophisticated, we ought to use it instead!).

Since we can fit a generalized additive model for a univariate time
series with missing time steps just fine, we can use this much more
sophisticated model to impute missing observations. Of course, this does
not help us fit the model, so we should do it only if the missing data
is of interest itself. Missing energy demand is just such a situation.
Perhaps we wish to report the total energy demand for the month, and
have one day of missing data. We ought to use the best imputation of
that data available: in this case, our model. We could further use the
uncertainty associated with that imputation to place bounds on the total
demand.

##### Anomaly detection

To detect anomalous behaviour effectively often requires a definition of
anomalous. There are many ways in which a time series can display
anomalous behaviour. For the particular use case of electricity demand
forecasting, we might desire to isolate incidents of unusually high or
low demand at the hourly level. Another possible definition of anomalous
behaviour could include a smaller excess demand sustained for several
hours or days. Anomaly detection comes down to identifying events that
are unlikely under the model. We could, for example, ask to be alerted
to any single observation that lies outside the 99% uncertainty interval
of forecast values (the region where 99% of samples lie).^[This is not the same
as a confidence interval, but is more useful in this case.] This
simple form of anomaly detection might be useful as a first attempt at
automatic identification of historical outliers.
