---
layout: model
title: Universal Sentence Encoder Multilingual Large
author: John Snow Labs
name: tfhub_use_multi_lg
date: 2020-12-08
task: Embeddings
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [embeddings, open_source, xx]
deprecated: true
annotator: UniversalSentenceEncoder
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks.

The model is trained and optimized for greater-than-word length text, such as sentences, phrases, or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is the variable-length text and the output is a 512-dimensional vector. The universal-sentence-encoder model has trained with a deep averaging network (DAN) encoder.

This model supports 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian) text encoder.

The details are described in the paper "[Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/abs/1907.04307)".

Note: This model only works on Linux and macOS operating systems and is not compatible with Windows due to the incompatibility of the SentencePiece library.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_multi_lg_xx_2.7.0_2.4_1607439900967.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tfhub_use_multi_lg_xx_2.7.0_2.4_1607439900967.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_multi_lg", "xx") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP', 'Me encanta usar SparkNLP']], ["text"]))
```
```scala
...
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_multi_lg", "xx")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("I love NLP", "Me encanta usar SparkNLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP", "Me encanta usar SparkNLP"]
embeddings_df = nlu.load('xx.use.multi_lg').predict(text, output_level='sentence')
embeddings_df
```

</div>


## Results

It gives a 512-dimensional vector of the sentences.

```bash
sentence	          xx_use_multi_lg_embeddings
	
0	I love NLP	          [-0.055500030517578125, 0.13188503682613373, -...
1	Me encanta usar SparkNLP  [0.008856392465531826, 0.06127211079001427, 0....
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_multi_lg|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|xx|

## Data Source

This embeddings model is imported from [https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)

## Benchmarking

- We apply this model to the STS benchmark for semantic similarity. The eval can be seen in the [example notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) made available. Results are shown below:

```bash
STSBenchmark                       | dev    | test  |
-----------------------------------|--------|-------|   
Correlation coefficient of Pearson | 0.837  | 0.825 |
```

- For semantic similarity retrieval, we evaluate the model on [Quora and AskUbuntu retrieval task.](https://arxiv.org/abs/1811.08008). Results are shown below:

```bash
Dataset                | Quora | AskUbuntu | Average |
-----------------------|-------|-----------|---------|
Mean Averge Precision  | 89.1  | 42.3      | 65.7    |
```

- For the translation pair retrieval, we evaluate the model on the United Nation Parallal Corpus. Results are shown below:

```bash
Language Pair  | en-es  | en-fr | en-ru | en-zh |
---------------|--------|-------|-------|-------|
Precision@1    | 86.1   | 83.3  | 88.9  | 78.8  |
```