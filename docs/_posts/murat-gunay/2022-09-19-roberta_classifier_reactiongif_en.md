---
layout: model
title: English RoBertaForSequenceClassification Cased model (from julien-c)
author: John Snow Labs
name: roberta_classifier_reactiongif
date: 2022-09-19
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
nav_key: models
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `reactiongif-roberta` is a English model originally trained by `julien-c`.

## Predicted Entities

`awww`, `agree`, `scared`, `oh_snap`, `hug`, `happy_dance`, `sorry`, `oops`, `yes`, `slow_clap`, `want`, `please`, `good_luck`, `sigh`, `thank_you`, `dance`, `high_five`, `you_got_this`, `yawn`, `thumbs_up`, `fist_bump`, `idk`, `shrug`, `ok`, `yolo`, `smh`, `shocked`, `facepalm`, `kiss`, `no`, `deal_with_it`, `hearts`, `applause`, `popcorn`, `thumbs_down`, `do_not_want`, `seriously`, `omg`, `eye_roll`, `wink`, `win`, `mic_drop`, `eww`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_reactiongif_en_4.1.0_3.0_1663609714083.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_reactiongif_en_4.1.0_3.0_1663609714083.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_reactiongif","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_reactiongif","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.roberta.by_julien_c").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_reactiongif|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/julien-c/reactiongif-roberta