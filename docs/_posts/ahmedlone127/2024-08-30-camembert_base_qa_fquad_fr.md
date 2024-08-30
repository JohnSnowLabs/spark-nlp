---
layout: model
title: French CamemBertForQuestionAnswering Base squadFR (camembert_base_qa_fquad)
author: John Snow Labs
name: camembert_base_qa_fquad
date: 2024-08-30
tags: [fr, french, question_answering, camembert, open_source, onnx, openvino]
task: Question Answering
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: openvino
annotator: BertForZeroShotClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `camembert_base_qa_fquad ` is a French model originally fine-tuned on a combo of three French Q&A datasets:

- PIAFv1.1
- FQuADv1.0
- SQuAD-FR (SQuAD automatically translated to French)

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_base_qa_fquad_fr_5.4.2_3.0_1725015714535.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_base_qa_fquad_fr_5.4.2_3.0_1725015714535.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Document_Assembler = MultiDocumentAssembler()\
     .setInputCols(["question", "context"])\
     .setOutputCols(["document_question", "document_context"])

Question_Answering = CamemBertForQuestionAnswering("camembert_base_qa_fquad","fr")\
     .setInputCols(["document_question", "document_context"])\
     .setOutputCol("answer")\
     .setCaseSensitive(True)

pipeline = Pipeline(stages=[Document_Assembler, Question_Answering])

data = spark.createDataFrame([["Où est-ce que je vis?","Mon nom est Wolfgang et je vis à Berlin."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)
```
```scala
val Document_Assembler = new MultiDocumentAssembler()
     .setInputCols(Array("question", "context"))
     .setOutputCols(Array("document_question", "document_context"))

val Question_Answering = CamemBertForQuestionAnswering("camembert_base_qa_fquad","fr")
     .setInputCols(Array("document_question", "document_context"))
     .setOutputCol("answer")
     .setCaseSensitive(True)

val pipeline = new Pipeline().setStages(Array(Document_Assembler, Question_Answering))

val data = Seq("Où est-ce que je vis?","Mon nom est Wolfgang et je vis à Berlin.").toDS.toDF("question", "context")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("fr.answer_question.camembert.fquad").predict("""Où est-ce que je vis?|||"Mon nom est Wolfgang et je vis à Berlin.""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_base_qa_fquad|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[label]|
|Language:|fr|
|Size:|667.9 MB|
|Case sensitive:|true|

## References

References

References

https://huggingface.co/etalab-ia/camembert-base-squadFR-fquad-piaf

## Benchmarking

```bash


{"f1": 80.61, "exact_match": 59.54}
```