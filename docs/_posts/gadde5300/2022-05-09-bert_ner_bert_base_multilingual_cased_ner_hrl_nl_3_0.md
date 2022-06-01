---
layout: model
title: Dutch Named Entity Recognition (from Davlan)
author: John Snow Labs
name: bert_ner_bert_base_multilingual_cased_ner_hrl
date: 2022-05-09
tags: [bert, ner, token_classification, nl, open_source]
task: Named Entity Recognition
language: nl
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `bert-base-multilingual-cased-ner-hrl` is a Dutch model orginally trained by `Davlan`.

## Predicted Entities

`LOC`, `DATE`, `PER`, `ORG`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_bert_base_multilingual_cased_ner_hrl_nl_3.4.2_3.0_1652099961577.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_multilingual_cased_ner_hrl","nl") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Ik hou van Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_bert_base_multilingual_cased_ner_hrl","nl") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Ik hou van Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_bert_base_multilingual_cased_ner_hrl|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|nl|
|Size:|665.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl
- https://camel.abudhabi.nyu.edu/anercorp/
- https://www.clips.uantwerpen.be/conll2003/ner/
- https://www.clips.uantwerpen.be/conll2003/ner/
- https://www.clips.uantwerpen.be/conll2002/ner/
- https://github.com/EuropeanaNewspapers/ner-corpora/tree/master/enp_FR.bnf.bio
- https://ontotext.fbk.eu/icab.html
- https://github.com/LUMII-AILab/FullStack/tree/master/NamedEntities
- https://www.clips.uantwerpen.be/conll2002/ner/
- https://github.com/davidsbatista/NER-datasets/tree/master/Portuguese