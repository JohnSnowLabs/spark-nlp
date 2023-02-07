---
layout: model
title: Sentence Entity Resolver for ICD-O (sbiobertresolve_icdo_augmented)
author: John Snow Labs
name: sbiobertresolve_icdo_augmented
date: 2022-06-06
tags: [licensed, clinical, en, icdo, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.5.2
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted clinical entities to ICD-O codes using `sbiobert_base_cased_mli` Sentence BERT Embeddings. Given an oncological entity found in the text (via NER models like `ner_jsl`), it returns top terms and resolutions along with the corresponding ICD-O codes to present more granularity with respect to body parts mentioned. It also returns the original `Topography` and `Histology`  codes, and their descriptions.

## Predicted Entities

`ICD-O Codes`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icdo_augmented_en_3.5.2_3.0_1654546345691.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icdo_augmented_en_3.5.2_3.0_1654546345691.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")\

ner_converter = NerConverterInternal()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")\
.setWhiteList(["Oncological"])

c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")\


resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icdo_augmented", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")\


resolver_pipeline = Pipeline(
stages = [
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
ner,
ner_converter,
c2doc,
sbert_embedder,
resolver
])

data = spark.createDataFrame([["""TRAF6 is a putative oncogene in a variety of cancers including  urothelial cancer , and malignant melanoma. WWP2 appears to regulate the expression of the well characterized tumor and tensin homolog (PTEN) in endometroid adenocarcinoma and squamous cell carcinoma."""]]).toDF("text")

result = resolver_pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverterInternal()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")
.setWhiteList(Array("Oncological"))

val c2doc = new Chunk2Doc()
.setInputCols("ner_chunk")
.setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols("ner_chunk_doc")
.setOutputCol("sentence_embeddings")


val resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icdo_augmented", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sentence_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val resolver_pipeline = new Pipeline().setStages(Array(document_assembler, 
sentenceDetectorDL, 
tokenizer, 
word_embeddings, 
ner, 
ner_converter,  
c2doc, 
sbert_embedder, 
resolver))

val data = Seq("""TRAF6 is a putative oncogene in a variety of cancers including  urothelial cancer , and malignant melanoma. WWP2 appears to regulate the expression of the well characterized tumor and tensin homolog (PTEN) in endometroid adenocarcinoma and squamous cell carcinoma.""").toDS.toDF("text")

val results = resolver_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icdo_augmented").predict("""TRAF6 is a putative oncogene in a variety of cancers including  urothelial cancer , and malignant melanoma. WWP2 appears to regulate the expression of the well characterized tumor and tensin homolog (PTEN) in endometroid adenocarcinoma and squamous cell carcinoma.""")
```

</div>

## Results

```bash
+--------------------------+-----------+---------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                     chunk|     entity|icdo_code|                                                          all_k_resolutions|                                                                all_k_codes|
+--------------------------+-----------+---------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                   cancers|Oncological|   8000/3|cancer:::carcinoma:::carcinomatosis:::neoplasms:::ceruminous carcinoma::...|8000/3:::8010/3:::8010/9:::800:::8420/3:::8140/3:::8010/3-C76.0:::8010/6...|
|         urothelial cancer|Oncological|   8120/3|urothelial carcinoma:::urothelial carcinoma in situ of urinary system:::...|8120/3:::8120/2-C68.9:::8010/3-C68.9:::8130/3-C68.9:::8070/3-C68.9:::813...|
|        malignant melanoma|Oncological|   8720/3|malignant melanoma:::malignant melanoma, of skin:::malignant melanoma, o...|8720/3:::8720/3-C44.9:::8720/3-C06.9:::8720/3-C69.9:::8721/3:::8720/3-C0...|
|                     tumor|Oncological|   8000/1|tumor:::tumorlet:::tumor cells:::askin tumor:::tumor, secondary:::pilar ...|8000/1:::8040/1:::8001/1:::9365/3:::8000/6:::8103/0:::9364/3:::8940/0:::...|
|endometroid adenocarcinoma|Oncological|   8380/3|endometrioid adenocarcinoma:::endometrioid adenoma:::scirrhous adenocarc...|8380/3:::8380/0:::8141/3-C54.1:::8560/3-C54.1:::8260/3-C54.1:::8380/3-C5...|
|   squamous cell carcinoma|Oncological|   8070/3|squamous cell carcinoma:::verrucous squamous cell carcinoma:::squamous c...|8070/3:::8051/3:::8070/2:::8052/3:::8070/3-C44.5:::8075/3:::8560/3:::807...|
+--------------------------+-----------+---------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icdo_augmented|
|Compatibility:|Healthcare NLP 3.5.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icdo_code]|
|Language:|en|
|Size:|175.7 MB|
|Case sensitive:|false|

## References

Trained on ICD-O Histology Behaviour dataset with sbiobert_base_cased_mli sentence embeddings. https://apps.who.int/iris/bitstream/handle/10665/96612/9789241548496_eng.pdf
