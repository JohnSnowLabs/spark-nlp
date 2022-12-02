---
layout: model
title: Google T5 for Closed Book Question Answering Small
author: John Snow Labs
name: google_t5_small_ssm_nq
date: 2021-01-08
task: Question Answering
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [open_source, t5, seq2seq, question_answering, en]
supported: true
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The model was pre-trained using T5's denoising objective on [C4](https://huggingface.co/datasets/c4), subsequently additionally pre-trained using REALM's salient span masking objective on Wikipedia, and finally fine-tuned on [Natural Questions (NQ)](https://huggingface.co/datasets/natural_questions).

Note: The model was fine-tuned on 100% of the train splits of Natural Questions (NQ) for 10k steps.

Other community Checkpoints: here

Paper: [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/1910.10683.pdf)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/google_t5_small_ssm_nq_en_2.7.1_2.4_1610137175322.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Either set the following tasks or have them inline with your input:

- nq question:
- trivia question:
- question:
- nq:

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("documents")

t5 = T5Transformer() \
.pretrained("google_t5_small_ssm_nq") \
.setTask("nq:")\
.setMaxOutputLength(200)\
.setInputCols(["documents"]) \
.setOutputCol("answer")

pipeline = Pipeline().setStages([document_assembler, t5])
results = pipeline.fit(data_df).transform(data_df)

results.select("answer.result").show(truncate=False)

```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("documents")

val t5 = T5Transformer
.pretrained("google_t5_small_ssm_nq")
.setTask("nq:")
.setInputCols(Array("documents"))
.setOutputCol("answer")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val model = pipeline.fit(dataDf)
val results = model.transform(dataDf)

results.select("answer.result").show(truncate = false)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.t5").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|google_t5_small_ssm_nq|
|Compatibility:|Spark NLP 2.7.1+|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[t5]|
|Language:|en|

## Data Source

The model was pre-trained using T5's denoising objective on C4, subsequently additionally pre-trained using REALM's salient span masking objective on Wikipedia, and finally fine-tuned on Natural Questions (NQ).