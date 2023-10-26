---
layout: model
title: BERT Embeddings (Large Cased)
author: John Snow Labs
name: bert_large_cased
date: 2023-10-24
tags: [open_source, embeddings, en, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_cased_en_5.1.4_3.4_1698191629229.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_large_cased_en_5.1.4_3.4_1698191629229.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = BertEmbeddings.pretrained("bert_large_cased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
result = pipeline_model.transform(spark.createDataFrame([['I love NLP']], ["text"]))
```
```scala
...
val embeddings = BertEmbeddings.pretrained("bert_large_cased", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
val data = Seq("I love NLP").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.bert.large_cased').predict(text, output_level='token')
embeddings_df
```
</div>

## Results

```bash

	token	en_embed_bert_large_cased_embeddings
	
	I	[-0.5893247723579407, -1.1389378309249878, -0....
	love	[-0.8002289533615112, -0.15043185651302338, 0....
	NLP	[-0.8995863199234009, 0.08327484875917435, 0.9...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_large_cased|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|