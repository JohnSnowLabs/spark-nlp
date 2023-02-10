---
layout: model
title: Korean ElectraForSequenceClassification Cased model (from searle-j)
author: John Snow Labs
name: electra_classifier_kote_for_easygoing_people
date: 2022-09-14
tags: [ko, open_source, electra, sequence_classification, classification]
task: Text Classification
language: ko
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ElectraForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `kote_for_easygoing_people` is a Korean model originally trained by `searle-j`.

## Predicted Entities

`깨달음`, `놀람`, `기쁨`, `부담/안_내킴`, `우쭐댐/무시함`, `공포/무서움`, `흐뭇함(귀여움/예쁨)`, `환영/호의`, `부끄러움`, `화남/분노`, `패배/자기혐오`, `귀찮음`, `짜증`, `불쌍함/연민`, `증오/혐오`, `기대감`, `안심/신뢰`, `행복`, `재미없음`, `절망`, `비장함`, `어이없음`, `지긋지긋`, `불평/불만`, `고마움`, `안타까움/실망`, `불안/걱정`, `즐거움/신남`, `한심함`, `뿌듯함`, `슬픔`, `죄책감`, `경악`, `없음`, `역겨움/징그러움`, `힘듦/지침`, `신기함/관심`, `편안/쾌적`, `당황/난처`, `의심/불신`, `감동/감탄`, `아껴주는`, `존경`, `서러움`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_classifier_kote_for_easygoing_people_ko_4.1.0_3.0_1663180008021.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_classifier_kote_for_easygoing_people_ko_4.1.0_3.0_1663180008021.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("electra_classifier_kote_for_easygoing_people","ko") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("electra_classifier_kote_for_easygoing_people","ko") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_classifier_kote_for_easygoing_people|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ko|
|Size:|467.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/searle-j/kote_for_easygoing_people