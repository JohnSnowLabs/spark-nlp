---
layout: model
title: Universal Sentence Encoder XLING Many
author: John Snow Labs
name: tfhub_use_xling_many
date: 2020-12-08
task: Embeddings
language: xx
edition: Spark NLP 2.7.0
spark_version: 2.4
deprecated: true
tags: [embeddings, open_source, xx]
supported: true
annotator: UniversalSentenceEncoder
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The Universal Sentence Encoder Cross-lingual (XLING) module is an extension of the Universal Sentence Encoder that includes training on multiple tasks across languages. The multi-task training setup is based on the paper "Learning Cross-lingual Sentence Representations via a Multi-task Dual Encoder".

This specific module is trained on English, French, German, Spanish, Italian, Chinese, Korean, and Japanese tasks, and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs.

It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box for a number of applications. The input to the module is variable length text in any of the eight aforementioned languages and the output is a 512 dimensional vector.

We note that one does not need to specify the language of the input, as the model was trained such that text across languages with similar meanings will have embeddings with high dot product scores.

Note: This model only works on Linux and macOS operating systems and is not compatible with Windows due to the incompatibility of the SentencePiece library.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_xling_many_xx_2.7.0_2.4_1607440840968.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tfhub_use_xling_many_xx_2.7.0_2.4_1607440840968.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_many", "xx") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP', 'Me encanta usar SparkNLP']], ["text"]))
```
```scala
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_many", "xx")
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("I love NLP", "Me encanta usar SparkNLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP", "Me encanta usar SparkNLP"]
embeddings_df = nlu.load('xx.use.xling_many').predict(text, output_level='sentence')
embeddings_df
```

</div>

## Results

It gives a 512-dimensional vector of the sentences.

```bash
        xx_use_xling_many_embeddings	                     sentence

0	[0.03621278703212738, 0.007045685313642025, -0...    I love NLP
1	[-0.0060035050846636295, 0.028749311342835426,...    Me encanta usar SparkNLP
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_xling_many|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|xx|

## Data Source

[https://tfhub.dev/google/universal-sentence-encoder-xling-many/1](https://tfhub.dev/google/universal-sentence-encoder-xling-many/1)
