---
layout: model
title: Extract Information from Termination Clauses (Md)
author: John Snow Labs
name: legner_termination_md
date: 2022-12-01
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole legal agreement. Instead:
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `legclf_termination_clause` Text Classifier to select only these paragraphs; 

This is a NER model which extracts information from Termination Clauses, like the subject (Who? Which party?) the action (verb) the object (What?) and the Indirect Object (to Whom?).

This is a `md` (medium version) of the classifier, trained with more data and being more resistent to false positives outside the specific section, which may help to run it at whole document level (although not recommended).

## Predicted Entities

`TERMINATION_SUBJECT`, `TERMINATION_ACTION`, `TERMINATION_OBJECT`, `TERMINATION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_termination_md_en_1.0.0_3.0_1669894724125.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_termination_md_en_1.0.0_3.0_1669894724125.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = legal.NerModel.pretrained('legner_termination_md','en','legal/models')\
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
|Model Name:|legner_termination_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.1 MB|

## References

In-house annotations of CUAD dataset.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-TERMINATION_INDIRECT_OBJECT	 4	 0	 6	 1.0	 0.4	 0.5714286
B-TERMINATION_INDIRECT_OBJECT	 3	 1	 4	 0.75	 0.42857143	 0.5454545
B-TERMINATION_OBJECT	 38	 22	 36	 0.6333333	 0.5135135	 0.5671642
I-TERMINATION_ACTION	 85	 27	 5	 0.7589286	 0.9444444	 0.8415842
I-TERMINATION_OBJECT	 294	 172	 294	 0.6309013	 0.5	 0.55787474
B-TERMINATION_SUBJECT	 37	 10	 8	 0.78723407	 0.82222223	 0.8043478
I-TERMINATION_SUBJECT	 26	 8	 7	 0.7647059	 0.7878788	 0.7761194
B-TERMINATION_ACTION	 36	 7	 5	 0.8372093	 0.8780488	 0.8571428
Macro-average	 523 247 365 0.770289 0.6593349 0.7105064
Micro-average	 523 247 365 0.6792208 0.588964 0.6308806
```