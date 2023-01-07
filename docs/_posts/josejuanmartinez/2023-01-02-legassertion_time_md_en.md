---
layout: model
title: Temporality / Certainty Assertion Status (md)
author: John Snow Labs
name: legassertion_time_md
date: 2023-01-02
tags: [time, possibility, en, licensed]
task: Assertion Status
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a medium (md) Assertion Status Model aimed to detect temporality (PRESENT, PAST, FUTURE) or Certainty (POSSIBLE) in your legal documents, which may improve the results of the `legassertion_time` (small) model.

## Predicted Entities

`PRESENT`, `PAST`, `FUTURE`, `POSSIBLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGASSERTION_TEMPORALITY){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legassertion_time_md_en_1.0.0_3.0_1672687111582.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings_ner")\

ner_model = legal.NerModel.pretrained('legner_contract_doc_parties', 'en', 'legal/models')\
    .setInputCols(["sentence", "token", "embeddings_ner"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setWhiteList(["DOC", "EFFDATE", "PARTY"]) # We will check time only on these

assertion = legal.AssertionDLModel.pretrained("legassertion_time_md", "en", "legal/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings "]) \
    .setOutputCol("assertion")


nlpPipeline = nlp.Pipeline(stages=[
            document_assembler, 
            sentence_detector,
            tokenizer,
            embeddings_ner,
            ner_model,
            ner_converter,
            assertion
            ])

empty_data = spark.createDataFrame([[""]]).toDF("text")  

model = nlpPipeline.fit(empty_data)

lp = LightPipeline(model)

texts = ["The subsidiaries of Atlantic Inc will participate in a merging operation",
    "The Conditions and Warranties of this agreement might be modified"]

lp.annotate(texts)
```

</div>

## Results

```bash
chunk,begin,end,entity_type,assertion
Atlantic Inc,20,31,ORG,FUTURE

chunk,begin,end,entity_type,assertion
Conditions and Warranties,4,28,DOC,POSSIBLE
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legassertion_time_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, doc_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

In-house annotations augmented with synthetic data.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
PRESENT	 115	 11	 5	 0.9126984	 0.9583333	 0.9349593
POSSIBLE	 79	 5	 4	 0.9404762	 0.9518072	 0.9461077
PAST	 54	 5	 11	 0.91525424	 0.83076924	 0.8709678
FUTURE	 77	 3	 4	 0.9625	 0.9506173	 0.95652175
Macro-average 325 24 24 0.9327322 0.9228818 0.92778087
Micro-average 325 24 24 0.9312321 0.9312321 0.9312321
```