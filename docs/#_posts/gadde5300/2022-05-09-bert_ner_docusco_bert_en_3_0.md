---
layout: model
title: English Named Entity Recognition (from browndw)
author: John Snow Labs
name: bert_ner_docusco_bert
date: 2022-05-09
tags: [bert, ner, token_classification, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition model, uploaded to Hugging Face, adapted and imported into Spark NLP. `docusco-bert` is a English model orginally trained by `browndw`.

## Predicted Entities

`Interactive`, `AcademicTerms`, `InformationChange`, `MetadiscourseCohesive`, `FirstPerson`, `InformationPlace`, `Updates`, `InformationChangeneritive`, `Reasoning`, `PublicTerms`, `Citation`, `Future`, `CitationHedged`, `InformationExnerition`, `Contingent`, `Strategic`, `PAD`, `CitationAuthority`, `Facilitate`, `Positive`, `ConfidenceHigh`, `InformationStates`, `AcademicWritingMoves`, `Uncertainty`, `SyntacticComplexity`, `Responsibility`, `Character`, `Narrative`, `MetadiscourseInteractive`, `InformationTopics`, `ConfidenceLow`, `ConfidenceHedged`, `ForceStressed`, `Negative`, `InformationChangeNegative`, `Description`, `Inquiry`, `InformationReportVerbs`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_docusco_bert_en_3.4.2_3.0_1652097061800.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = BertForTokenClassification.pretrained("bert_ner_docusco_bert","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

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

val tokenClassifier = BertForTokenClassification.pretrained("bert_ner_docusco_bert","en") 
    .setInputCols(Array("sentence", "token")) 
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector, tokenizer, tokenClassifier))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_docusco_bert|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/browndw/docusco-bert
- https://www.english-corpora.org/coca/
- https://www.cmu.edu/dietrich/english/research-and-publications/docuscope.html
- https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=docuscope&btnG=
- https://graphics.cs.wisc.edu/WP/vep/2017/02/14/guest-post-data-mining-king-lear/
- https://journals.sagepub.com/doi/full/10.1177/2055207619844865
- https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
- https://www.english-corpora.org/coca/
- https://arxiv.org/pdf/1810.04805