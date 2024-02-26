---
layout: model
title: English wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa CamemBertForQuestionAnswering from cstorm125
author: John Snow Labs
name: wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa
date: 2024-01-21
tags: [camembert, en, open_source, question_answering, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForQuestionAnswering model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa` is a English model originally trained by cstorm125.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa_en_5.2.4_3.0_1705871878194.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa_en_5.2.4_3.0_1705871878194.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


document_assembler = MultiDocumentAssembler() \
    .setInputCol(["question", "context"]) \
    .setOutputCol(["document_question", "document_context"])
    
    
spanClassifier = CamemBertForQuestionAnswering.pretrained("wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa","en") \
            .setInputCols(["document_question","document_context"]) \
            .setOutputCol("answer")

pipeline = Pipeline().setStages([document_assembler, spanClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val document_assembler = new MultiDocumentAssembler()
    .setInputCol(Array("question", "context")) 
    .setOutputCol(Array("document_question", "document_context"))
    
val spanClassifier = CamemBertForQuestionAnswering  
    .pretrained("wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa", "en")
    .setInputCols(Array("document_question","document_context")) 
    .setOutputCol("answer") 

val pipeline = new Pipeline().setStages(Array(document_assembler, spanClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wangchanberta_base_wiki_20210520_news_spm_span_mask_finetune_qa|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|392.0 MB|

## References

https://huggingface.co/cstorm125/wangchanberta-base-wiki-20210520-news-spm_span-mask-finetune-qa