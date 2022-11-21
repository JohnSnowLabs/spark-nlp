---
layout: model
title: T5 Clinical Summarization / QA model
author: John Snow Labs
name: t5_base_mediqa_mnli
date: 2021-02-19
tags: [t5, licensed, clinical, en]
supported: true
recommended: true
task: Summarization
language: en
edition: Healthcare NLP 2.7.4
spark_version: 2.4
annotator: T5Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The T5 transformer model described in the seminal paper “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer” can perform a variety of tasks, such as text summarization, question answering and translation. More details about using the model can be found in the paper (https://arxiv.org/pdf/1910.10683.pdf). This model is specifically trained on medical data for text summarization and question answering.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/t5_base_mediqa_mnli_en_2.7.4_2.4_1613750257481.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("documents")

sentence_detector = SentenceDetectorDLModel().pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
.setInputCols("documents")\
.setOutputCol("sentence")

t5 = T5Transformer().pretrained("t5_base_mediqa_mnli", "en", "clinical/models") \
.setInputCols(["sentence"]) \
.setOutputCol("t5_output")\
.setTask("summarize medical questions:")\
.setMaxOutputLength(200)

pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
t5
])
pipeline = Pipeline(stages=[
document_assembler, 
sentence_detector,
t5
])
data = spark.createDataFrame([
[1, "content:SUBJECT: Normal physical traits but no period MESSAGE: I'm a 40 yr. old woman that has infantile reproductive organs and have never experienced a mensus. I have had Doctors look but they all say I just have infantile female reproductive organs. When I try to look for answers on the internet I cannot find anything. ALL my \"girly\" parts are normal. My organs never matured. Could you give me more information please. focus:all"]
]).toDF('id', 'text')
results = pipeline.fit(data).transform(data)
results.select("t5_output.result").show(truncate=False)

```



{:.nlu-block}
```python
import nlu
nlu.load("en.t5.mediqa").predict("""content:SUBJECT: Normal physical traits but no period MESSAGE: I'm a 40 yr. old woman that has infantile reproductive organs and have never experienced a mensus. I have had Doctors look but they all say I just have infantile female reproductive organs. When I try to look for answers on the internet I cannot find anything. ALL my \""")
```

</div>

## Results

```bash
What are the treatments for mensus?, What are the treatments for infantile female reproductive organs?, What are the treatments for cancer?, What are the treatments for organ transplantation?, What are the treatments for cancer?, What are the treatments for cancer?
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_base_mediqa_mnli|
|Compatibility:|Healthcare NLP 2.7.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|

## Data Source

Trained on MEDIQA2021 and MedNLI Datasets
