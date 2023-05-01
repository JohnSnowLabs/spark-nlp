---
layout: model
title: English DistilBertForSequenceClassification Base Uncased model
author: John Snow Labs
name: distilbert_sequence_classifier_distilbert_base_uncased_finetuned_sst_2_english
date: 2022-08-23
tags: [distilbert, sequence_classification, open_source, en]
task: Text Classification
language: en
nav_key: models
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilbert-base-uncased-finetuned-sst-2-english` is a English model originally trained by HuggingFace.

## Predicted Entities

`POSITIVE`, `NEGATIVE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_base_uncased_finetuned_sst_2_english_en_4.1.0_3.0_1661277744844.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_distilbert_base_uncased_finetuned_sst_2_english_en_4.1.0_3.0_1661277744844.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_base_uncased_finetuned_sst_2_english","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_distilbert_base_uncased_finetuned_sst_2_english","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distil_bert.glue.uncased_base_finetuned").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_distilbert_base_uncased_finetuned_sst_2_english|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|249.7 MB|
|Case sensitive:|false|
|Max sentence length:|128|

## References

- https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
- https://paperswithcode.com/sota?task=Text+Classification&dataset=glue
- https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb
- https://twitter.com/aureliengeron