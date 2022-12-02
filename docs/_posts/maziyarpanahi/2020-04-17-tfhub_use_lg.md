---
layout: model
title: Universal Sentence Encoder Large
author: John Snow Labs
name: tfhub_use_lg
date: 2020-04-17
task: Embeddings
language: en
edition: Spark NLP 2.4.0
spark_version: 2.4
tags: [embeddings, en, open_source]
supported: true
annotator: UniversalSentenceEncoder
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
The Universal Sentence Encoder encodes text into high-dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.

The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity, and the results can be seen in the example notebook made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.

The details are described in the paper "[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tfhub_use_lg_en_2.4.0_2.4_1587136993894.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_lg", "en") \
.setInputCols("document") \
.setOutputCol("sentence_embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP', 'Many thanks']], ["text"]))
```

```scala
...
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use_lg", "en")
.setInputCols("document")
.setOutputCol("sentence_embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, embeddings))
val data = Seq("I love NLP", "Many thanks").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP", "Many thanks"]
embeddings_df = nlu.load('en.embed_sentence.tfhub_use.lg').predict(text, output_level='sentence')
embeddings_df
```

</div>

{:.h2_title}
## Results
```bash
en_embed_sentence_tfhub_use_lg_embeddings	        sentence
		
0	[0.05463508144021034, 0.013395714573562145, 0....	I love NLP
1	[0.03631748631596565, 0.006253095343708992, 0....	Many thanks
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tfhub_use_lg|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.4.0|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[sentence_embeddings]|
|Language:|[en]|
|Dimension:|512|
|Case sensitive:|true|


{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/universal-sentence-encoder-large/3](https://tfhub.dev/google/universal-sentence-encoder-large/3)
