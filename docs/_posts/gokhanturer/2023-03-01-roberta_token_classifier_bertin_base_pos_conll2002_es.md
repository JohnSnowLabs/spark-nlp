---
layout: model
title: Spanish RobertaForTokenClassification Base Cased model (from bertin-project)
author: John Snow Labs
name: roberta_token_classifier_bertin_base_pos_conll2002
date: 2023-03-01
tags: [es, open_source, roberta, token_classification, ner, tensorflow]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bertin-base-pos-conll2002-es` is a Spanish model originally trained by `bertin-project`.

## Predicted Entities

`DA`, `VAM`, `I`, `VSM`, `PP`, `VSS`, `DI`, `AQ`, `Y`, `VMN`, `Fit`, `Fg`, `Fia`, `Fpa`, `Fat`, `VSN`, `Fpt`, `DD`, `VAP`, `SP`, `NP`, `Fh`, `VAI`, `CC`, `Fd`, `VMG`, `NC`, `PX`, `DE`, `Fz`, `PN`, `Fx`, `Faa`, `Fs`, `Fe`, `VSP`, `DP`, `VAS`, `VSG`, `PT`, `Ft`, `VAN`, `PI`, `P0`, `RG`, `RN`, `CS`, `DN`, `VMI`, `Fp`, `Fc`, `PR`, `VSI`, `AO`, `VMM`, `PD`, `VMS`, `DT`, `Z`, `VMP`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_bertin_base_pos_conll2002_es_4.3.0_3.0_1677703697571.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_bertin_base_pos_conll2002_es_4.3.0_3.0_1677703697571.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_bertin_base_pos_conll2002","es") \
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
 
val tokenClassifier = RobertaForTokenClassification.pretrained("roberta_token_classifier_bertin_base_pos_conll2002","es") 
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
|Model Name:|roberta_token_classifier_bertin_base_pos_conll2002|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|426.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/bertin-project/bertin-base-pos-conll2002-es