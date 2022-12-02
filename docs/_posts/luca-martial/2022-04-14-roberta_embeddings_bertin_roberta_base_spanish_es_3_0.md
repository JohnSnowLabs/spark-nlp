---
layout: model
title: Spanish RoBERTa Embeddings (Bertin Base)
author: John Snow Labs
name: roberta_embeddings_bertin_roberta_base_spanish
date: 2022-04-14
tags: [roberta, embeddings, es, open_source]
task: Embeddings
language: es
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
recommended: true
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model for Spanish Language, trained within the Bertin project. Other non-base Bertin models can be found [here](https://nlp.johnsnowlabs.com/models?q=bertin). The model was uploaded to Hugging Face, adapted and imported into Spark NLP. `bertin-roberta-base-spanish` is a Spanish model orginally trained by `bertin-project`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_bertin_roberta_base_spanish_es_3.4.2_3.0_1649945200032.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_bertin_roberta_base_spanish","es") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Me encanta chispa nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_bertin_roberta_base_spanish","es") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Me encanta chispa nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.embed.bertin_roberta_base_spanish").predict("""Me encanta chispa nlp""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_bertin_roberta_base_spanish|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|es|
|Size:|234.7 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/bertin-project/bertin-roberta-base-spanish
- https://github.com/google/flax
- https://en.wikipedia.org/wiki/List_of_languages_by_total_number_of_speakers
- https://arxiv.org/pdf/2107.07253.pdf
- https://arxiv.org/abs/1907.11692
- https://www.wired.com/2017/04/courts-using-ai-sentence-criminals-must-stop-now/
- https://www.washingtonpost.com/technology/2019/05/16/police-have-used-celebrity-lookalikes-distorted-images-boost-facial-recognition-results-research-finds/
- https://www.wired.com/story/ai-college-exam-proctors-surveillance/
- https://www.eff.org/deeplinks/2020/09/students-are-pushing-back-against-proctoring-surveillance-apps
- https://www.washingtonpost.com/technology/2019/10/22/ai-hiring-face-scanning-algorithm-increasingly-decides-whether-you-deserve-job/
- https://www.technologyreview.com/2021/07/21/1029860/disability-rights-employment-discrimination-ai-hiring/
- https://www.insider.com/china-is-testing-ai-recognition-on-the-uighurs-bbc-2021-5
- https://www.health.harvard.edu/blog/anti-asian-racism-breaking-through-stereotypes-and-silence-2021041522414
- https://discord.com/channels/858019234139602994/859113060068229190
