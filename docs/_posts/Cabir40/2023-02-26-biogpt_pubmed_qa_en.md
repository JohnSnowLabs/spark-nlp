---
layout: model
title: Medical Question Answering (biogpt)
author: John Snow Labs
name: biogpt_pubmed_qa
date: 2023-02-26
tags: [licensed, en, clinical, biogpt, gpt, pubmed, question_answering, tensorflow]
task: Question Answering
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MedicalQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model has been trained with medical documents and can generate two types of answers, short and long.
Types of questions are supported: "`short"` (producing yes/no/maybe) answers and `"full"` (long answers).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/biogpt_pubmed_qa_en_4.3.0_3.0_1677406773484.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/biogpt_pubmed_qa_en_4.3.0_3.0_1677406773484.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = MultiDocumentAssembler()\
    .setInputCols("question", "context")\
    .setOutputCols("document_question", "document_context")

med_qa = MedicalQuestionAnswering.pretrained("medical_qa_biogpt","en","clinical/models")\
    .setInputCols(["document_question", "document_context"])\
    .setMaxNewTokens(100)\
    .setOutputCol("answer")\
    .setQuestionType("short") #long

pipeline = Pipeline(stages=[document_assembler, med_qa])

paper_abstract = "The visual indexing theory proposed by Zenon Pylyshyn (Cognition, 32, 65-97, 1989) predicts that visual attention mechanisms are employed when mental images are projected onto a visual scene. Recent eye-tracking studies have supported this hypothesis by showing that people tend to look at empty places where requested information has been previously presented. However, it has remained unclear to what extent this behavior is related to memory performance. The aim of the present study was to explore whether the manipulation of spatial attention can facilitate memory retrieval. In two experiments, participants were asked first to memorize a set of four objects and then to determine whether a probe word referred to any of the objects. The results of both experiments indicate that memory accuracy is not affected by the current focus of attention and that all the effects of directing attention to specific locations on response times can be explained in terms of stimulus-stimulus and stimulus-response spatial compatibility."
long_question = "What is the effect of directing attention on memory?"
yes_no_question = "Does directing attention improve memory for items?"

data = spark.createDataFrame(
    [
        [long_question, paper_abstract, "long"],
        [yes_no_question, paper_abstract, "short"],
    ]
).toDF("question", "context", "question_type")


pipeline.fit(data).transform(data.where("question_type == 'long'"))\
    .select("answer.result")\
    .show(truncate=False)

pipeline.fit(data).transform(data.where("question_type == 'short'"))\
    .select("answer.result")\
    .show(truncate=False)
```
```scala
val document_assembler = new MultiDocumentAssembler()
    .setInputCols("question", "context")
    .setOutputCols("document_question", "document_context")

val med_qa = MedicalQuestionAnswering.pretrained("medical_qa_biogpt","en","clinical/models")
    .setInputCols(Array("document_question", "document_context"))
    .setMaxNewTokens(100)
    .setOutputCol("answer")
    .setQuestionType("short") #long

val pipeline = new Pipeline().setStages(Array(document_assembler, med_qa))

paper_abstract = "The visual indexing theory proposed by Zenon Pylyshyn (Cognition, 32, 65-97, 1989) predicts that visual attention mechanisms are employed when mental images are projected onto a visual scene. Recent eye-tracking studies have supported this hypothesis by showing that people tend to look at empty places where requested information has been previously presented. However, it has remained unclear to what extent this behavior is related to memory performance. The aim of the present study was to explore whether the manipulation of spatial attention can facilitate memory retrieval. In two experiments, participants were asked first to memorize a set of four objects and then to determine whether a probe word referred to any of the objects. The results of both experiments indicate that memory accuracy is not affected by the current focus of attention and that all the effects of directing attention to specific locations on response times can be explained in terms of stimulus-stimulus and stimulus-response spatial compatibility."
long_question = "What is the effect of directing attention on memory?"
yes_no_question = "Does directing attention improve memory for items?"

val data = Seq( 
    (long_question, paper_abstract,"long" ),
    (yes_no_question, paper_abstract, "short"))
    .toDS.toDF("question", "context", "question_type")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
#short result
+------+
|result|
+------+
|[no]  |
+------+

#long result
+------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                |
+------------------------------------------------------------------------------------------------------------------------------------------------------+
|[the results of the two experiments suggest that the visual indexeing theory does not fully explain the effects that spatial attention has on memory.]|
+------------------------------------------------------------------------------------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biogpt_pubmed_qa|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|
|Case sensitive:|true|