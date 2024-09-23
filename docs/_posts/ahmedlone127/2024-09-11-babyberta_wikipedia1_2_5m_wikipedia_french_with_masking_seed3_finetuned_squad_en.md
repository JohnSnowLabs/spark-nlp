---
layout: model
title: English babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad RoBertaForQuestionAnswering from lielbin
author: John Snow Labs
name: babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad
date: 2024-09-11
tags: [en, open_source, onnx, question_answering, roberta]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad` is a English model originally trained by lielbin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad_en_5.5.0_3.0_1726039179363.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad_en_5.5.0_3.0_1726039179363.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
             
documentAssembler = MultiDocumentAssembler() \
     .setInputCol(["question", "context"]) \
     .setOutputCol(["document_question", "document_context"])
    
spanClassifier = RoBertaForQuestionAnswering.pretrained("babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad","en") \
     .setInputCols(["document_question","document_context"]) \
     .setOutputCol("answer")

pipeline = Pipeline().setStages([documentAssembler, spanClassifier])
data = spark.createDataFrame([["What framework do I use?","I use spark-nlp."]]).toDF("document_question", "document_context")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new MultiDocumentAssembler()
    .setInputCol(Array("question", "context")) 
    .setOutputCol(Array("document_question", "document_context"))
    
val spanClassifier = RoBertaForQuestionAnswering.pretrained("babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad", "en")
    .setInputCols(Array("document_question","document_context")) 
    .setOutputCol("answer") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, spanClassifier))
val data = Seq("What framework do I use?","I use spark-nlp.").toDS.toDF("document_question", "document_context")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|babyberta_wikipedia1_2_5m_wikipedia_french_with_masking_seed3_finetuned_squad|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|32.0 MB|

## References

https://huggingface.co/lielbin/BabyBERTa-wikipedia1_2.5M_wikipedia_french-with-Masking-seed3-finetuned-SQuAD