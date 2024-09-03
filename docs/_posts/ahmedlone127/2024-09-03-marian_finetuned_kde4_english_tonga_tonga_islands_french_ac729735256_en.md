---
layout: model
title: English marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256 MarianTransformer from ac729735256
author: John Snow Labs
name: marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256
date: 2024-09-03
tags: [en, open_source, onnx, translation, marian]
task: Translation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: MarianTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256` is a English model originally trained by ac729735256.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256_en_5.5.0_3.0_1725403804011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256_en_5.5.0_3.0_1725403804011.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

sentenceDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
      .setInputCols(["document"]) \
      .setOutputCol("translation")

marian = MarianTransformer.pretrained("marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256","en") \
      .setInputCols(["sentence"]) \
      .setOutputCol("embeddings")       
        
pipeline = Pipeline().setStages([documentAssembler, sentenceDL, marian])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val marian = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")

val embeddings = MarianTransformer.pretrained("marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256","en") 
    .setInputCols(Array("sentence")) 
    .setOutputCol("translation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDL, marian))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marian_finetuned_kde4_english_tonga_tonga_islands_french_ac729735256|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentences]|
|Output Labels:|[translation]|
|Language:|en|
|Size:|508.2 MB|

## References

https://huggingface.co/ac729735256/marian-finetuned-kde4-en-to-fr