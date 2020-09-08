+++
title = "Notes: The Transformer"
date = "2020-09-07"
+++

The transformer, described in "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" is a network architecture capable of sequence-to-sequence translations, without using recurrence, and thus, allowing parallelization.

<!--more-->

## The Transformer

In order to solve sequence transduction or sequence-to-sequence
problems (i.e. from $(x_1, x_2, ..., x_n)$ to $(y_1, y_n, ..., y_m)$, we can use an encoder followed by a decoder.

As classic recurrent (RNN) encoder-decoder networks, the Transformer is made of an encoder, connected to the decoder. In  "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)", 6 encoder layers are stacked together, followed by 6 other decoder layer.

An encoder layer is made of a self-attention component followed by a feed-forward NN.
> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values [...]. [1]


### Credits
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)