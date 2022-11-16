---
layout: model
title: Vietnamese DistilBERT Base Cased Embeddings
author: John Snow Labs
name: distilbert_base_cased
date: 2022-01-13
tags: [embeddings, distilbert, vietnamese, vi, open_source]
task: Embeddings
language: vi
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: DistilBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This embeddings model was imported from `Hugging Face`. It's a custom version of `distilbert_base_multilingual_cased` and it gives the same representations produced by the original model which preserves the original accuracy.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_cased_vi_3.3.4_3.0_1642064850307.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

distilbert = DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi")\
.setInputCols(["sentence",'token'])\
.setOutputCol("embeddings")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, distilbert])

text = """Tôi yêu Spark NLP"""

data = spark.createDataFrame([[text]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
...

val embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "vi")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("Tôi yêu Spark NLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("vi.embed.distilbert.cased").predict("""Tôi yêu Spark NLP""")
```

</div>

## Results

```bash
+-----+--------------------+
|token|          embeddings|
+-----+--------------------+
|  Tôi|[-0.38760236, -0....|
|  yêu|[-0.3357051, -0.5...|
|Spark|[-0.20642707, -0....|
|  NLP|[-0.013280544, -0...|
+-----+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_cased|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|vi|
|Size:|211.6 MB|
|Case sensitive:|false|
