---
layout: model
title: Roberta Clinical Word Embeddings (Spanish)
author: John Snow Labs
name: roberta_base_biomedical
date: 2021-11-01
tags: [embeddings, spanish, biomedical, clinical, roberta, es]
task: Embeddings
language: es
edition: Spark NLP 3.3.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Biomedical pretrained language model for Spanish with a `768 embeddings dimension`, imported from Hugging Face (https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-es) to be used in SparkNLP using RobertaEmbeddings() transfformer class.

This model is a RoBERTa-based model trained on a biomedical corpus in Spanish collected from several sources (see dataset section). The training corpus has been tokenized using a byte version of Byte-Pair Encoding (BPE) used in the original RoBERTA model with a vocabulary size of 52,000 tokens. The pretraining consists of a masked language model training at the subword level following the approach employed for the RoBERTa base model with the same hyperparameters as in the original work. The training lasted a total of 48 hours with 16 NVIDIA V100 GPUs of 16GB DDRAM, using Adam optimizer with a peak learning rate of 0.0005 and an effective batch size of 2,048 sentences.

To see more details, please check the official page in Hugging Face: https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-es

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_biomedical_es_3.3.0_3.0_1635781845226.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_biomedical_es_3.3.0_3.0_1635781845226.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
.setInputCol("term")\
.setOutputCol("document")

tokenizer = nlp.Tokenizer()\
.setInputCols("document")\
.setOutputCol("token")

roberta_embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
.setInputCols(["document", "token"])\
.setOutputCol("roberta_embeddings")

pipeline = Pipeline(stages = [
documentAssembler,
tokenizer,
roberta_embeddings])
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("term")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val roberta_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")
.setInputCols(Array("document", "token"))
.setOutputCol("roberta_embeddings")

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
tokenizer,
roberta_embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("es.embed.roberta_base_biomedical").predict("""Put your text here.""")
```

</div>

## Results

```bash
The model has been evaluated on the Named Entity Recognition (NER) using the following datasets (taken from https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es)

* PharmaCoNER: is a track on chemical and drug mention recognition from Spanish medical texts (for more info see: https://temu.bsc.es/pharmaconer/).

* CANTEMIST: is a shared task specifically focusing on named entity recognition of tumor morphology, in Spanish (for more info see: https://zenodo.org/record/3978041#.YTt5qH2xXbQ).

* ICTUSnet: consists of 1,006 hospital discharge reports of patients admitted for stroke from 18 different Spanish hospitals. It contains more than 79,000 annotations for 51 different kinds of variables.
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_biomedical|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[embeddings]|
|Language:|es|

## Data Source

Datasets are available in the official author(s) github project, available here: https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es, and include:

* Medical crawler	745,705,946	Crawler of more than 3,000 URLs belonging to Spanish biomedical and health domains.

* Clinical cases misc.	102,855,267	A miscellany of medical content, essentially clinical cases. Note that a clinical case report is a scientific publication where medical practitioners share patient cases and it is different from a clinical note or document.

* Clinical notes/documents	91,250,080	Collection of more than 278K clinical documents, including discharge reports, clinical course notes and X-ray reports, for a total of 91M tokens.

* Scielo	60,007,289	Publications written in Spanish crawled from the Spanish SciELO server in 2017.

* BARR2_background	24,516,442	Biomedical Abbreviation Recognition and Resolution (BARR2) containing Spanish clinical case study sections from a variety of clinical disciplines.

* Wikipedia_life_sciences	13,890,501	Wikipedia articles crawled 04/01/2021 with the Wikipedia API python library starting from the "Ciencias_de_la_vida" category up to a maximum of 5 subcategories. Multiple links to the same articles are then discarded to avoid repeating content.

* Patents	13,463,387	Google Patent in Medical Domain for Spain (Spanish). The accepted codes (Medical Domain) for Json files of patents are: "A61B", "A61C","A61F", "A61H", "A61K", "A61L","A61M", "A61B", "A61P".

* EMEA	5,377,448	Spanish-side documents extracted from parallel corpora made out of PDF documents from the European Medicines Agency.

* mespen_Medline	4,166,077	Spanish-side articles extracted from a collection of Spanish-English parallel corpus consisting of biomedical scientific literature. The collection of parallel resources are aggregated from the MedlinePlus source.

* PubMed	1,858,966	Open-access articles from the PubMed repository crawled in 2017.

## Benchmarking

```bash
Taken from https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es:

Task/models   F1 | Precision | Recall
PharmaCoNER   90.04 | 88.92 | 91.18
CANTEMIST     83.34 | 81.48 | 85.30
ICTUSnet      88.08 | 84.92 | 91.50
```
