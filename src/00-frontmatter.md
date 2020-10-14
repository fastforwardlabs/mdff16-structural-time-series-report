# Structural Time Series

FF16 &middot; _October 2020_

![Structural Time-Series report cover](figures/ff16-cover-splash.png)

_This is an applied research report by <a href="https://www.cloudera.com/products/fast-forward-labs-research.html">Cloudera Fast Forward</a>.
We write reports about emerging technologies.
Accompanying each report are working prototypes or code that exhibits the capabilities of the algorithm and offer detailed technical advice on its practical application.
Read our full report on structural time series below or <a href="/FF16-Structural_Time_Series-Cloudera_Fast_Forward.pdf" target="_blank" id="report-pdf-download">download the PDF</a>._

[[TOC]]

## Introduction

Time series data is ubiquitous, and many methods of processing and modeling data over time have been developed.
As with any data science project, there is no one method to rule them all, and the most appropriate approach ought to depend on the data in question, the goals of the modeler, and the time and resources available.

We have often encountered time series problems in our client-facing work at Cloudera Fast Forward.
These come in a variety of shapes and sizes.
Sometimes there are millions of parallel time series, where external factors are expected to be extremely predictive, and there is a single well-defined metric to optimize.
At the other end of the spectrum is a single, noisy time series, with many missing points and no additional information.
In this report, we will focus on approaches that are more suited to the latter case.
In particular, we will investigate structural time series, which are especially useful in cases where the time series exhibits some periodic patterns.

We will describe a family of models—Generalized Additive Models, or GAMs—which have the advantage of being scalable and easy to interpret, and tend to perform well on a large number of time series problems.
We’ll first describe what we mean by a structural time series and generalized additive models.
Then, we’ll cover some considerations for model construction and evaluation.
Finally, we’ll walk through a practical application, in which we’ll forecast the demand for electricity in California.
(This report is accompanied by [the code](https://github.com/fastforwardlabs/structural-time-series) for that application.)
