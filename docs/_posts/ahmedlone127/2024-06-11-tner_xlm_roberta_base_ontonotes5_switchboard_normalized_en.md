---
layout: model
title: English tner_xlm_roberta_base_ontonotes5_switchboard_normalized XlmRoBertaForTokenClassification from anonymoussubmissions
author: John Snow Labs
name: tner_xlm_roberta_base_ontonotes5_switchboard_normalized
date: 2024-06-11
tags: [en, open_source, onnx, token_classification, xlm_roberta, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tner_xlm_roberta_base_ontonotes5_switchboard_normalized` is a English model originally trained by anonymoussubmissions.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tner_xlm_roberta_base_ontonotes5_switchboard_normalized_en_5.4.0_3.0_1718132044391.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tner_xlm_roberta_base_ontonotes5_switchboard_normalized_en_5.4.0_3.0_1718132044391.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier  = XlmRoBertaForTokenClassification.pretrained("tner_xlm_roberta_base_ontonotes5_switchboard_normalized","en") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("tner_xlm_roberta_base_ontonotes5_switchboard_normalized", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tner_xlm_roberta_base_ontonotes5_switchboard_normalized|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|858.7 MB|

## References

https://huggingface.co/anonymoussubmissions/tner-xlm-roberta-base-ontonotes5-switchboard-normalized