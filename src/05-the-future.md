## The future

Generalized additive models provide a flexible, interpretable, and
broadly applicable basis for time series modeling, and can at the least
provide an improved baseline for many time series problems. Prophet
provides a mature and robust tool for the use of GAMs, and due to its
success, we expect it to continue improving and becoming more flexible
over time. Optimistically, we also hope to see the development of
similar toolkits for other well-scoped problems.

### Ethical considerations

Cynthia Rudin's excellent paper,
[Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/abs/1811.10154),
advocates for using models that are themselves interpretable, instead of
explaining black box models with additional models and approximations,
especially when the model informs a high stakes decision;
the prediction of criminal recidivism is given as an example.
Since generalized additive models are
a simple sum of components with known properties like periodicity or
piecewise linearity, *they are inherently interpretable* in a way that
methods which highly entangle their features (such as random forests or
neural networks) are not. Because each component may be non-linear, they
simultaneously provide a substantial increase in modeling flexibility
over linear models.

The paper highlights that, in contrast to commonly held belief, there is
not *always* a tradeoff between accuracy and interpretability.
[Intelligible Models for Classification and Regression](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf)
quantifies this for generalized additive models, with a thorough
empirical comparison between various GAM fitting methods and ensembles
of trees. While the tree ensembles ultimately obtain the lowest error
rates on most of the studied datasets, they do not always, and often the
error rate is within one standard deviation of the closest GAM method
(where the mean and variance for the error rates are calculated with
cross-validation). As such, even when accuracy is the goal, it is often
worth first pursuing an interpretable model, such as a GAM.

As we demonstrate with our simple forecasting app, capturing uncertainty
in a forecast unlocks novel capabilities, like asking probabilistic
questions. Moreover, being explicit about uncertainty is responsible
data science practice.

### Further research

Generalized additive models are not new, and are far from the only
approach to solving time series problems. For univariate time series,
[The Automatic Statistician](http://www.automaticstatistician.com/research/) and
related work in probabilistic programming (such as
[Time Series Structure Discovery via Probabilistic Program Synthesis](https://arxiv.org/abs/1611.07051))
are particularly promising research directions, due to their efforts to automate
the exploration of structural components.

Forecasting a single time series is an age-old problem, but the advent
of the data age has exposed new shapes of time series problems. It is
not uncommon to seek a forecast of hundreds, thousands, or even millions
of concurrent time series, which necessitates a different approach. The
recent success of transformer models for natural language processing is
spurring work on applying attention-based architectures to discrete time
series, as in
[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/abs/1703.07015),
[Temporal Pattern Attention for Multivariate Time Series Forecasting](https://arxiv.org/abs/1809.04206),
and [Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case](https://arxiv.org/abs/2001.08317).
Such models seem well suited
to automating time series forecasting for highly multivariate time
series, though means for explaining the predictions of transformer
models is an open research frontier.

Time series forecasting is perennially relevant, and while many
established methodologies exist, the space does not lack for innovation.
We look forward to seeing what the future holds, as new methods and
tools develop.

### Resources

The primary reference for Prophet is the paper,
[Forecasting at Scale](https://peerj.com/preprints/3190/).

If you're interested in this structural approach to time series, you may
be interested in probabilistic methods in general. In particular, we
recommend Richard McElreath's
[*Statistical Rethinking*](https://xcelab.net/rm/statistical-rethinking/), which does
an excellent job of pedagogy, building Bayesian methods from simple
intuitive foundations. (His lectures on the book are excellent as well,
and you can find them via
[his YouTube channel](https://www.youtube.com/channel/UCNJK6_DZvcMqNSzQdEkzvzA/videos)).
If you are more inclined to code (though we should note that
*Statistical Rethinking* strongly emphasizes doing code exercises), you
may also like
[*Bayesian Methods for Hackers*](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/).

For more on general forecasting, the go-to reference is
[*Forecasting: Principles and Practice*](https://otexts.com/fpp2/).
