## Introduction

Old federated learning content

To train a machine learning model you generally need to move all the data to a
single machine or, failing that, to a cluster of machines in a data center.
This can be difficult for two reasons. 

First, there can be privacy barriers. A smartphone user may not want to share
their baby photos with an application developer. A user of industrial equipment
may not want to share sensor data with the manufacturer or a competitor. And
healthcare providers are not totally free to share their patients' data with
drug companies.

Second, there are practical engineering challenges. A huge amount of valuable
training data is created on hardware at the edges of slow and unreliable
networks, such as smartphones, IoT devices, or equipment in far-flung industrial 
facilities such as mines and oil rigs. Communication with such devices can be 
slow and expensive.

This research report and its associated prototype introduce _federated
learning_, an algorithmic solution to these problems.

![In federated learning, a network of nodes shares models rather than training data with a server.](figures/ff09-23.png)

In federated learning, a server coordinates a network of nodes, each of which
has training data that it cannot or will not share directly. The nodes each
train a model, and it is that model which they share with the server. By not
transferring the data itself, federated learning helps to ensure privacy and
minimizes communication costs.

In this report, we discuss use cases ranging from smartphones to web browsers
to healthcare to corporate IT to video analytics. Our working prototype focuses
in particular on industrial predictive maintenance with IoT data, where training
data is a sensitive asset.

