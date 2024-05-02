---
layout: model
title: Sentence Entity Resolver for UMLS CUI Codes (Clinical Drug)
author: John Snow Labs
name: sbiobertresolve_umls_clinical_drugs
date: 2024-05-02
tags: [entity_resolution, licensed, clinical, en, open_source]
task: Entity Resolution
language: en
edition: Spark NLP 5.3.2
spark_version: 3.2
supported: true
annotator: SentenceEntityResolverModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities to UMLS CUI codes. It is trained on 2022AA UMLS dataset. The complete dataset has 127 different categories, and this model is trained on the Clinical Drug category using sbiobert_base_cased_mli embeddings.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sbiobertresolve_umls_clinical_drugs_en_5.3.2_3.2_1714666834404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sbiobertresolve_umls_clinical_drugs_en_5.3.2_3.2_1714666834404.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

sentence_detector = SentenceDetector()\
      .setInputCols(["document"])\
      .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols("sentence")\
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("posology_ner")

ner_model_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "posology_ner"])\
    .setOutputCol("posology_ner_chunk")\
    .setWhiteList(["DRUG"])

chunk2doc = Chunk2Doc().setInputCols("posology_ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")\
     .setCaseSensitive(False)

resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_umls_clinical_drugs", "en", "clinical/models") \
     .setInputCols(["sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

pipeline = Pipeline(stages = [document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_model_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["""She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg."""]]).toDF("text")

results = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentence_detector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

val tokenizer = new Tokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel
      .pretrained("embeddings_clinical", "en", "clinical/models")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")

val ner_model = MedicalNerModel
      .pretrained("ner_posology_greedy", "en", "clinical/models")
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("posology_ner")

val ner_model_converter = new NerConverterInternal()
      .setInputCols(Array("sentence", "token", "posology_ner"))
      .setOutputCol("ner_chunk")
      .setWhiteList("DRUG")

val chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings
      .pretrained("sbiobert_base_cased_mli", "en","clinical/models")
      .setInputCols(Array("ner_chunk_doc"))
      .setOutputCol("sbert_embeddings")
      .setCaseSensitive(False)
    
val resolver = SentenceEntityResolverModel
      .pretrained("sbiobertresolve_umls_clinical_drugs", "en", "clinical/models") 
      .setInputCols(Array("ner_chunk_doc", "sbert_embeddings")) 
      .setOutputCol("resolution")
      .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner_model, ner_model_converter, chunk2doc, sbert_embedder, resolver))
    
val data = Seq("She was immediately given hydrogen peroxide 30 mg to treat the infection on her leg, and has been advised Neosporin Cream for 5 days. She has a history of taking magnesium hydroxide 100mg/1ml and metformin 1000 mg.").toDF("text") 

val res = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------------+------+---------+--------------------------+------------------------------------------------------------+------------------------------------------------------------+
|                    ner_chunk|entity|umls_code|               description|                                               all_k_results|                                           all_k_resolutions|
+-----------------------------+------+---------+--------------------------+------------------------------------------------------------+------------------------------------------------------------+
|      hydrogen peroxide 30 mg|  DRUG| C1126248|hydrogen peroxide 30 mg/ml|C1126248:::C0304655:::C1605252:::C0304656:::C1154260:::C2...|hydrogen peroxide 30 mg/ml:::hydrogen peroxide solution 3...|
|              Neosporin Cream|  DRUG| C0132149|           neosporin cream|C0132149:::C4722788:::C0704071:::C0698988:::C1252084:::C3...|neosporin cream:::neomycin sulfate cream:::neosporin topi...|
|magnesium hydroxide 100mg/1ml|  DRUG| C1134402|magnesium hydroxide 100 mg|C1134402:::C1126785:::C4317023:::C4051486:::C4047137:::C1...|magnesium hydroxide 100 mg:::magnesium hydroxide 100 mg/m...|
|            metformin 1000 mg|  DRUG| C0987664|         metformin 1000 mg|C0987664:::C2719784:::C0978482:::C2719786:::C4282269:::C2...|metformin 1000 mg:::metformin hydrochloride 1000 mg:::met...|
+-----------------------------+------+---------+--------------------------+------------------------------------------------------------+------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_umls_clinical_drugs|
|Compatibility:|Spark NLP 5.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[umls_code]|
|Language:|en|
|Size:|1.9 GB|
|Case sensitive:|false|

## References

Trained on 2022AA UMLS dataset’s Clinical Drug category. https://www.nlm.nih.gov/research/umls/index.html