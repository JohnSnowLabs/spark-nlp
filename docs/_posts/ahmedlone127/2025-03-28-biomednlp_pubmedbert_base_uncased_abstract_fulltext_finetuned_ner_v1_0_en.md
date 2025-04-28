---
layout: model
title: English biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0 BertForTokenClassification from mevol
author: John Snow Labs
name: biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0
date: 2025-03-28
tags: [en, open_source, onnx, token_classification, bert, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0` is a English model originally trained by mevol.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0_en_5.5.1_3.0_1743137275502.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0_en_5.5.1_3.0_1743137275502.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier  = BertForTokenClassification.pretrained("biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0","en") \
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

val tokenClassifier = BertForTokenClassification.pretrained("biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0", "en")
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
|Model Name:|biomednlp_pubmedbert_base_uncased_abstract_fulltext_finetuned_ner_v1_0|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|408.3 MB|

## References

https://huggingface.co/mevol/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-ner_v1.0