---
layout: model
title: English RobertaForTokenClassification Base Cased model (from philschmid)
author: John Snow Labs
name: roberta_ner_distilroberta_base_ner_wikiann_conll2003_4_class
date: 2022-08-13
tags: [bert, ner, open_source, en]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilroberta-base-ner-wikiann-conll2003-4-class` is a English model originally trained by `philschmid`.

## Predicted Entities

`PER`, `MISC`, `ORG`, `LOC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_ner_distilroberta_base_ner_wikiann_conll2003_4_class_en_4.1.0_3.0_1660405008509.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_ner_distilroberta_base_ner_wikiann_conll2003_4_class_en_4.1.0_3.0_1660405008509.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("roberta_ner_distilroberta_base_ner_wikiann_conll2003_4_class","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("roberta_ner_distilroberta_base_ner_wikiann_conll2003_4_class","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.distil_roberta.conll_wikiann.distilled_base.by_philschmid").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_ner_distilroberta_base_ner_wikiann_conll2003_4_class|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|306.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/philschmid/distilroberta-base-ner-wikiann-conll2003-4-class
- https://paperswithcode.com/sota?task=Token+Classification&dataset=wikiann-conll2003