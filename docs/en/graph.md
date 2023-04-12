---
layout: docs
header: true
seotitle:
title: Spark NLP - Tensorflow Graph
permalink: /docs/en/graph
key: docs-graph
modify_date: "2020-02-03"
---

NER DL uses Char CNNs - BiLSTM - CRF Neural Network architecture. Spark NLP defines this architecture through a Tensorflow graph, which requires the following parameters:

- Tags
- Embeddings Dimension
- Number of Chars

Spark NLP infers these values from the training dataset used in [NerDLApproach annotator](annotators.md#ner-dl) and tries to load the graph embedded on spark-nlp package.
Currently, Spark NLP has graphs for the most common combination of tags, embeddings, and number of chars values:

{:.table-model-big.w7}
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

*Graph [parameter] should be [value]: Could not find a suitable tensorflow graph for embeddings dim: [value] tags: [value] nChars: [value]. Check https://sparknlp.org/docs/en/graph for instructions to generate the required graph.*

To overcome this exception message we have to follow these steps:

1. Clone [spark-nlp github repo](https://github.com/JohnSnowLabs/spark-nlp)

2. Run python file `create_models` with number of tags, embeddings dimension and number of char values mentioned on your exception message error.

    ```bash
    cd spark-nlp/python/tensorflow
    export PYTHONPATH=lib/ner
    python ner/create_models.py [number_of_tags] [embeddings_dimension] [number_of_chars] [output_path]
    ```

3. This will generate a graph on the directory defined on `output_path argument.

4. Retry training with `NerDLApproach` annotator but this time use the parameter `setGraphFolder` with the path of your graph.

**Note:**  Make sure that you have Python 3 and Tensorflow 1.15.0 installed on your system since `create_models` requires those versions to generate the graph successfully.

**Note:**  We also have a notebook in the same directory if you prefer Jupyter notebook to cerate your custom graph (`create_models.ipynb`).
