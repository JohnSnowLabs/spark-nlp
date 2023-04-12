---
layout: model
title: Igbo Named Entity Recognition (from mbeukman)
author: John Snow Labs
name: xlmroberta_ner_xlm_roberta_base_finetuned_swahili_finetuned_ner_igbo
date: 2022-05-17
tags: [xlm_roberta, ner, token_classification, ig, open_source]
task: Named Entity Recognition
language: ig
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: XlmRoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `xlm-roberta-base-finetuned-swahili-finetuned-ner-igbo` is a Igbo model orginally trained by `mbeukman`.

## Predicted Entities

`PER`, `ORG`, `LOC`, `DATE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_xlm_roberta_base_finetuned_swahili_finetuned_ner_igbo_ig_3.4.2_3.0_1652809139166.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_xlm_roberta_base_finetuned_swahili_finetuned_ner_igbo_ig_3.4.2_3.0_1652809139166.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_xlm_roberta_base_finetuned_swahili_finetuned_ner_igbo","ig") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["Ahụrụ m n'anya na-atọ m ụtọ"]]).toDF("text")

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

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlmroberta_ner_xlm_roberta_base_finetuned_swahili_finetuned_ner_igbo","ig") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("Ahụrụ m n'anya na-atọ m ụtọ").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_xlm_roberta_base_finetuned_swahili_finetuned_ner_igbo|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|ig|
|Size:|1.0 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-swahili-finetuned-ner-igbo
- https://arxiv.org/abs/2103.11811
- https://github.com/Michael-Beukman/NERTransfer
- https://www.apache.org/licenses/LICENSE-2.0
- https://github.com/Michael-Beukman/
