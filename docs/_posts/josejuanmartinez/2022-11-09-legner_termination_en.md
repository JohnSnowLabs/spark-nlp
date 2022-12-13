---
layout: model
title: Extract Information from Termination Clauses (sm)
author: John Snow Labs
name: legner_termination
date: 2022-11-09
tags: [termination, en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_termination_clause` Text Classifier to select only these paragraphs; 

This is a NER model which extracts information from Termination Clauses, like the subject (Who? Which party?) the action (verb) the object (What?) and the Indirect Object (to Whom?).

## Predicted Entities

`TERMINATION_SUBJECT`, `TERMINATION_ACTION`, `TERMINATION_OBJECT`, `TERMINATION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_termination_en_1.0.0_3.0_1667988803376.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_termination_en_1.0.0_3.0_1667988803376.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_termination','en','legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

text = "(b) Either Party may terminate this Agreement"
data = spark.createDataFrame([[test]]).toDF("text")
model = nlpPipeline.fit(data)
```

</div>

## Results

```bash
+-----------+---------------------+
|      token|            ner_label|
+-----------+---------------------+
|          (|                    O|
|          b|                    O|
|          )|                    O|
|     Either|B-TERMINATION_SUBJECT|
|      Party|I-TERMINATION_SUBJECT|
|        may| B-TERMINATION_ACTION|
|  terminate| I-TERMINATION_ACTION|
|       this|                    O|
|  Agreement|                    O|
+-----------+---------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_termination|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.4 MB|

## References

In-house annotations of CUAD dataset.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-TERMINATION_INDIRECT_OBJECT	 6	 2	 0	 0.75	 1.0	 0.85714287
B-TERMINATION_INDIRECT_OBJECT	 5	 3	 2	 0.625	 0.71428573	 0.6666667
B-TERMINATION_OBJECT	 48	 13	 25	 0.78688526	 0.65753424	 0.7164179
I-TERMINATION_ACTION	 84	 11	 12	 0.8842105	 0.875	 0.8795811
I-TERMINATION_OBJECT	 337	 75	 145	 0.81796116	 0.6991701	 0.75391495
B-TERMINATION_SUBJECT	 43	 5	 1	 0.8958333	 0.97727275	 0.9347826
I-TERMINATION_SUBJECT	 38	 6	 0	 0.8636364	 1.0	 0.9268293
B-TERMINATION_ACTION	 42	 4	 1	 0.9130435	 0.9767442	 0.94382024
Macro-average	                 -        -        -    0.8170713     0.86250085  0.83917177
Micro-average	                 -        -        -     0.83518004   0.76425856  0.7981469
```
