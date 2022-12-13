---
layout: model
title: Legal Indemnification NER (Light, md)
author: John Snow Labs
name: legner_indemnifications_md
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
- Use the `legclf_indemnification_clause` Text Classifier to select only these paragraphs; 

This is a Legal Named Entity Recognition Model to identify the Subject (who), Action (web), Object(the indemnification) and Indirect Object (to whom) from Indemnification clauses.

This is a `md` (medium version) of the classifier, trained with more data and being more resistent to false positives outside the specific section, which may help to run it at whole document level (although not recommended).

## Predicted Entities

`INDEMNIFICATION`, `INDEMNIFICATION_SUBJECT`, `INDEMNIFICATION_ACTION`, `INDEMNIFICATION_INDIRECT_OBJECT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_indemnifications_md_en_1.0.0_3.0_1669894326703.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_indemnifications_md_en_1.0.0_3.0_1669894326703.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
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

ner_model = legal.NerModel.pretrained('legner_indemnifications_md', 'en', 'legal/models')\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[documentAssembler,sentenceDetector,tokenizer,embeddings,ner_model,ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text='''The Company shall protect and indemnify the Supplier against any damages, losses or costs whatsoever'''

data = spark.createDataFrame([[text]]).toDF("text")
model = nlpPipeline.fit(data)
lmodel = LightPipeline(model)
res = lmodel.annotate(text)
```

</div>

## Results

```bash
+----------+---------------------------------+
|     token|                        ner_label|
+----------+---------------------------------+
|       The|                                O|
|   Company|                                O|
|     shall|         B-INDEMNIFICATION_ACTION|
|   protect|         I-INDEMNIFICATION_ACTION|
|       and|                                O|
| indemnify|         B-INDEMNIFICATION_ACTION|
|       the|                                O|
|  Supplier|B-INDEMNIFICATION_INDIRECT_OBJECT|
|   against|                                O|
|       any|                                O|
|   damages|                B-INDEMNIFICATION|
|         ,|                                O|
|    losses|                B-INDEMNIFICATION|
|        or|                                O|
|     costs|                B-INDEMNIFICATION|
|whatsoever|                                O|
+----------+---------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_indemnifications_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house annotated examples from CUAD legal dataset

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-INDEMNIFICATION_ACTION	 9	 2	 0	 0.8181818	 1.0	 0.90000004
B-INDEMNIFICATION_INDIRECT_OBJECT	 24	 7	 0	 0.7741935	 1.0	 0.8727273
B-INDEMNIFICATION_SUBJECT	 5	 2	 0	 0.71428573	 1.0	 0.8333334
I-INDEMNIFICATION_SUBJECT	 3	 0	 0	 1.0	 1.0	 1.0
B-INDEMNIFICATION	 23	 2	 0	 0.92	 1.0	 0.9583333
I-INDEMNIFICATION_INDIRECT_OBJECT	 9	 3	 2	 0.75	 0.8181818	 0.78260875
B-INDEMNIFICATION_ACTION	 9	 4	 0	 0.6923077	 1.0	 0.8181818
I-INDEMNIFICATION	 5	 5	 0	 0.5	 1.0	 0.6666667
Macro-average	 87 25 2 0.77112114 0.97727275  0.8620434
Micro-average	 87 25 2 0.77678573 0.9775281 0.86567163
```