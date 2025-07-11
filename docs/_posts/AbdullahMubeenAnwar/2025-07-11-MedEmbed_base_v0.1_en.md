---
layout: model
title: "MedEmbed: Specialized Embedding Model for Medical and Clinical Information Retrieval"
author: John Snow Labs
name: MedEmbed_base_v0.1
date: 2025-07-11
tags: [bert, medembed, medical_embedding, clinical_embedding, information_retrieval, en, open_source, openvino]
task: Embeddings
language: en
edition: Spark NLP 6.0.4
spark_version: 3.4
supported: true
engine: openvino
annotator: BertSentenceEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

MedEmbed is a family of embedding models fine-tuned specifically for medical and clinical data, designed to enhance performance in healthcare-related natural language processing (NLP) tasks, particularly information retrieval.

This model is intended for use in medical and clinical contexts to improve information retrieval, question answering, and semantic search tasks. It can be integrated into healthcare systems, research tools, and medical literature databases to enhance search capabilities and information access.

Technical Blog Post: https://huggingface.co/blog/abhinand/medembed-finetuned-embedding-models-for-medical-ir

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/MedEmbed_base_v0.1_en_6.0.4_3.4_1752193265660.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/MedEmbed_base_v0.1_en_6.0.4_3.4_1752193265660.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BertSentenceEmbeddings 
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

Bert = BertSentenceEmbeddings.load("MedEmbed_base_v0.1") \
    .setInputCols(["document"]) \
    .setOutputCol("Bert")

pipeline = Pipeline(stages=[
    document_assembler,
    Bert
])

data = spark.createDataFrame([[
    "William Henry Gates III (born October 28, 1955) is an American business magnate, "
    "software developer, investor, and philanthropist."
]]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.selectExpr("explode(Bert.embeddings) as embeddings").show()
```

</div>

## Results

```bash
+--------------------+
|          embeddings|
+--------------------+
|[-0.17034447, 0.3...|
+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|MedEmbed_base_v0.1|
|Compatibility:|Spark NLP 6.0.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|390.6 MB|
|Case sensitive:|true|

## References

https://huggingface.co/datasets/abhinand/MedEmbed-training-triplets-v1

## Sample text from the training dataset

How did the patient's condition respond to the treatment?

neg: The patient's symptoms resolved after receiving treatment and undergoing follow-up echocardiograms.
pos: The patient responded positively to the prescribed treatment of diethylcarbamazine. No further follow-up is required.

## Benchmarking

```bash
MedEmbed consistently outperforms general-purpose embedding models across various medical NLP benchmarks:

ArguAna
MedicalQARetrieval
NFCorpus
PublicHealthQA
TRECCOVID
```