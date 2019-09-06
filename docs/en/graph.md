---
layout: article
title: Tensorflow Graph
permalink: /docs/en/graph
key: docs-graph
modify_date: "2019-09-06"
---

NER DL uses Char CNNs - BiLSTM - CRF Neural Network architecture. Spark NLP defines this architecture through a Tensorflow graph, which requires the following parameters:

- Tags
- Embeddings Dimension
- Number of Chars

Spark NLP infers these values from the training dataset used in [NerDLApproach annotator](annotators.md#ner-dl) and tries to load the graph embedded on spark-nlp package.
Currently, Spark NLP has graphs for the most common combination of tags, embeddings, and number of chars values:

| Tags | Embeddings Dimension |
| :--- | :------------------: | 
|  10  |       100            |
|  10  |       200            |
|  10  |       300            |
|  10  |       768            |
|  10  |       1024           |
|  25  |       300            |

All of these graphs use an LSTM of size 128 and number of chars 100

In case, your train dataset has a different number of tags, embeddings dimension, number of chars and LSTM size combinations shown in the table above, `NerDLApproach` will raise an **IllegalArgumentException** exception during runtime with the message below:

*Graph [parameter] should be [value]: Could not find a suitable tensorflow graph for embeddings dim: [value] tags: [value] nChars: [value]. Generate graph by python code in python/tensorflow/ner/create_models before usage and use setGraphFolder Param to point to output.*

To overcome this exception message we have to follow these steps:

1. Clone [spark-nlp github repo](https://github.com/JohnSnowLabs/spark-nlp)
2. Go to python/tensorflow/ner/ path and start a jupyter notebook

3. Open `create_models`  notebook

4. Make sure on the last cell that `create_graph` function is set with embeddings dimension, tags and number of char values mentioned on your exception message error.

5. The notebook generates a graph on the same directory of `create_models`. You can move it to another local directory if you want.

6. Retry training with `NerDLApproach` annotator but this time use the parameter `setGraphFolder` with the path of your graph.

**Note:**  Make sure that you have Python 3 and Tensorflow 1.12.0 installed on your system since `create_models` notebook requires those versions to generate the graph successfully
