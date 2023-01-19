---
layout: model
title: Detect Cellular/Molecular Biology Entities
author: John Snow Labs
name: ner_cellular_en
date: 2020-04-22
task: Named Entity Recognition
language: en
edition: Healthcare NLP 2.4.2
spark_version: 2.4
tags: [ner, en, clinical, licensed]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained named entity recognition deep learning model for molecular biology related terms. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

## Predicted Entities 
`DNA`, `Cell_type`, `Cell_line`, `RNA`, `Protein`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CELLULAR/){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_cellular_en_2.4.2_2.4_1587513308751.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_cellular_en_2.4.2_2.4_1587513308751.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")
cellular_ner = NerDLModel.pretrained("ner_cellular", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, cellular_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive. ']], ["text"]))

```

```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val celular_ner = NerDLModel.pretrained("ner_cellular", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, cellular_ner, ner_converter))

val data = Seq("Detection of various other intracellular signaling proteins is also described. Genetic characterization of transactivation of the human T-cell leukemia virus type 1 promoter: Binding of Tax to Tax-responsive element 1 is mediated by the cyclic AMP-responsive members of the CREB/ATF family of transcription factors. To achieve a better understanding of the mechanism of transactivation by Tax of human T-cell leukemia virus type 1 Tax-responsive element 1 (TRE-1), we developed a genetic approach with Saccharomyces cerevisiae. We constructed a yeast reporter strain containing the lacZ gene under the control of the CYC1 promoter associated with three copies of TRE-1. Expression of either the cyclic AMP response element-binding protein (CREB) or CREB fused to the GAL4 activation domain (GAD) in this strain did not modify the expression of the reporter gene. Tax alone was also inactive.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.

```bash
|chunk                                                      |ner      |
+-----------------------------------------------------------+---------+
|intracellular signaling proteins                           |protein  |
|human T-cell leukemia virus type 1 promoter                |DNA      |
|Tax                                                        |protein  |
|Tax-responsive element 1                                   |DNA      |
|cyclic AMP-responsive members                              |protein  |
|CREB/ATF family                                            |protein  |
|transcription factors                                      |protein  |
|Tax                                                        |protein  |
|human T-cell leukemia virus type 1 Tax-responsive element 1|DNA      |
|TRE-1),                                                    |DNA      |
|lacZ gene                                                  |DNA      |
|CYC1 promoter                                              |DNA      |
|TRE-1                                                      |DNA      |
|cyclic AMP response element-binding protein                |protein  |
|CREB                                                       |protein  |
|CREB                                                       |protein  |
|GAL4 activation domain                                     |protein  |
|GAD                                                        |protein  |
|reporter gene                                              |DNA      |
|Tax                                                        |protein  |
+-----------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_cellular|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.2|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on the JNLPBA corpus containing more than 2.404 publication abstracts with ``'embeddings_clinical'``.
http://www.geniaproject.org/

{:.h2_title}
## Benchmarking
```bash
|    | label         |     tp |    fp |   fn |     prec |      rec |       f1 |
|---:|:--------------|-------:|------:|-----:|---------:|---------:|---------:|
|  0 | B-cell_line   |    377 |   203 |  123 | 0.65     | 0.754    | 0.698148 |
|  1 | I-DNA         |   1519 |   277 |  266 | 0.845768 | 0.85098  | 0.848366 |
|  2 | I-protein     |   3981 |   911 |  786 | 0.813778 | 0.835116 | 0.824309 |
|  3 | B-protein     |   4483 |  1433 |  579 | 0.757776 | 0.885618 | 0.816724 |
|  4 | I-cell_line   |    786 |   340 |  203 | 0.698046 | 0.794742 | 0.743262 |
|  5 | I-RNA         |    178 |    42 |    9 | 0.809091 | 0.951872 | 0.874693 |
|  6 | B-RNA         |     99 |    28 |   19 | 0.779528 | 0.838983 | 0.808163 |
|  7 | B-cell_type   |   1440 |   294 |  480 | 0.83045  | 0.75     | 0.788177 |
|  8 | I-cell_type   |   2431 |   377 |  559 | 0.865741 | 0.813044 | 0.838565 |
|  9 | B-DNA         |    814 |   267 |  240 | 0.753006 | 0.772296 | 0.762529 |
| 10 | Macro-average | 16108  | 4172  | 3264 | 0.780318 | 0.824665 | 0.801879 |
| 11 | Micro-average | 16108  | 4172  | 3264 | 0.79428  | 0.831509 | 0.812469 |
```
