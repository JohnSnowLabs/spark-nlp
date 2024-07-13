---
layout: model
title: English T5ForConditionalGeneration Base Cased model (from mrm8488)
author: John Snow Labs
name: t5_base_finetuned_summarize_news
date: 2024-07-13
tags: [en, open_source, t5, onnx]
task: Text Generation
language: en
edition: Spark NLP 5.4.1
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `t5-base-finetuned-summarize-news` is a English model originally trained by `mrm8488`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_base_finetuned_summarize_news_en_5.4.1_3.0_1720882684717.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_base_finetuned_summarize_news_en_5.4.1_3.0_1720882684717.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols("text") \
    .setOutputCols("document")

t5 = T5Transformer.pretrained("t5_base_finetuned_summarize_news","en") \
    .setInputCols("document") \
    .setOutputCol("answers")
    
pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols("text")
      .setOutputCols("document")
       
val t5 = T5Transformer.pretrained("t5_base_finetuned_summarize_news","en") 
    .setInputCols("document")
    .setOutputCol("answers")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_base_finetuned_summarize_news|
|Compatibility:|Spark NLP 5.4.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|1.0 GB|

## References

References

- https://huggingface.co/mrm8488/t5-base-finetuned-summarize-news
- https://github.com/abhimishra91
- https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
- https://www.kaggle.com/sunnysai12345/news-summary
- https://arxiv.org/pdf/1910.10683.pdf
- https://i.imgur.com/jVFMMWR.png
- https://www.kaggle.com/sunnysai12345/news-summary
- https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
- https://github.com/abhimishra91
- https://twitter.com/mrm8488
- https://www.linkedin.com/in/manuel-romero-cs/