---
layout: model
title: Universal Sentence Encoder XLING English and Spanish
author: John Snow Labs
name: tfhub_use_xling_en_es
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

This specific module is trained on English and Spanish (en-es) tasks, and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. 

It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box for a number of applications. The input to the module is variable length English or Spanish text and the output is a 512 dimensional vector. 

We note that one does not need to specify the language that the input is in, as the model was trained such that English and Spanish text with similar meanings will have similar (high dot product score) embeddings. We also note that this model can be used for monolingual English (and potentially monolingual Spanish) tasks with comparable or even better performance than the purely English Universal Sentence Encoder.

Note: This model only works on Linux and macOS operating systems and is not compatible with Windows due to the incompatibility of the SentencePiece library.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_xling_en_es_xx_2.7.0_2.4_1607440558771.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_en_es", "xx") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP', 'Me encanta usar SparkNLP']], ["text"]))
```
```scala
...
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_xling_en_es", "xx")
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
embeddings_df = nlu.load('xx.use.xling_en_es').predict(text, output_level='sentence')
embeddings_df
```

</div>

## Results

It gives a 512-dimensional vector of the sentences.

```bash
        xx_use_xling_en_es_embeddings	                     sentence
 
0	[-0.02727784588932991, 0.022969702258706093, 0...    I love NLP
1	[-0.01980777457356453, 0.03035994991660118, 0....    Me encanta usar SparkNLP
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_xling_en_es|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|xx|

## Data Source

This embeddings model is imported from [https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1](https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1)
