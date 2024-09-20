---
layout: model
title: English finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search DistilBertForTokenClassification from benjaminzwhite
author: John Snow Labs
name: finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search
date: 2024-09-01
tags: [en, open_source, onnx, token_classification, distilbert, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search` is a English model originally trained by benjaminzwhite.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search_en_5.4.2_3.0_1725171385865.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search_en_5.4.2_3.0_1725171385865.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier  = DistilBertForTokenClassification.pretrained("finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search","en") \
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

val tokenClassifier = DistilBertForTokenClassification.pretrained("finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search", "en")
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
|Model Name:|finetuned_distilbert_base_uncased_conll2003_test_after_optuna_hp_search|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/benjaminzwhite/finetuned-distilbert-base-uncased-conll2003-test-after-Optuna-HP-search