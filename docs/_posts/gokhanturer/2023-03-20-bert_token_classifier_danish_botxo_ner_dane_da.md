---
layout: model
title: Danish BertForTokenClassification Cased model (from Maltehb)
author: John Snow Labs
name: bert_token_classifier_danish_botxo_ner_dane
date: 2023-03-20
tags: [da, open_source, bert, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: da
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `danish-bert-botxo-ner-dane` is a Danish model originally trained by `Maltehb`.

## Predicted Entities

`ORG`, `PER`, `[CLS]`, `[SEP]`, `LOC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_danish_botxo_ner_dane_da_4.3.1_3.0_1679332757159.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_danish_botxo_ner_dane_da_4.3.1_3.0_1679332757159.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_danish_botxo_ner_dane","da") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

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
 
val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_danish_botxo_ner_dane","da") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_danish_botxo_ner_dane|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|da|
|Size:|412.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Maltehb/danish-bert-botxo-ner-dane
- #danish-bert-version-2-uncased-by-certainlyhttpscertainlyio-previously-known-as-botxo-finetuned-for-named-entity-recognition-on-the-dane-datasethttpsdanlpalexandradk304bd159d5dedatasetsddtzip-hvingelby-et-al-2020-by-malte-højmark-bertelsen
- https://certainly.io/
- https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip
- https://certainly.io/
- https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip
- https://certainly.io/
- https://github.com/certainlyio/nordic_bert
- https://www.certainly.io/blog/danish-bert-model/
- https://www.dropbox.com/s/19cjaoqvv2jicq9/danish_bert_uncased_v2.zip?dl=1
- https://github.com/botxo/nordic_bert
- https://www.aclweb.org/anthology/2020.lrec-1.565
- https://twitter.com/malteH_B
- https://www.linkedin.com/in/malte-h%C3%B8jmark-bertelsen-9a618017b/
- https://www.instagram.com/maltemusen/