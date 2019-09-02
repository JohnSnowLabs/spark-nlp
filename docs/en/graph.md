---
layout: article
title: Evaluation
permalink: /docs/en/graph
key: docs-graph
modify_date: "2019-09-02"
---

## Tensorflow Graph

NER DL uses Char CNNs - BiLSTM - CRF Neural Network architecture. Thus, while training it uses a Tensorflow graph, which requires the following parameters:
- Tags
- Embeddings Dimension
- Number of Chars

Spark NLP infers these values from the training dataset used in [NERDLApproach annotator](annotators.md#ner-dl) and tries to load the graph embedded on spark-nlp package.
Currently Spark NLP has graphs for the most common combination of tags, embeddings and number of chars values:

| Tags | Embeddings Dimension |
| :--- | :------------------: | 
|  10  |       100            |
|  10  |       200            |
|  10  |       300            |
|  10  |       768            |
|  10  |       1024           |
|  25  |       300            |

All of these graphs use an LSTM of size 128 and number of chars 100

In case, your train dataset has a different number of tags, embeddings dimension, number of chars and LSTM size combinations show in the table above. NER DL approach will raise an **IllegalArgumentException** exception during runtime with the message below:
*Graph [parameter] should be [value]: Could not find a suitable tensorflow graph for embeddings dim: [value] tags: [value] nChars: [value]. Generate graph by python code in python/tensorflow/ner/create_models before usage and use setGraphFolder Param to point to output.*
