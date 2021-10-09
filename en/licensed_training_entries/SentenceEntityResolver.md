{%- capture title -%}
SentenceEntityResolver
{%- endcapture -%}

{%- capture description -%}
Contains all the parameters and methods to train a SentenceEntityResolverModel.
The model transforms a dataset with Input Annotation type SENTENCE_EMBEDDINGS, coming from e.g.
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings)
and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please use SentenceEntityResolverModel
and see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
ENTITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Training a SNOMED resolution model using BERT sentence embeddings
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
documentAssembler = DocumentAssembler() \
  .setInputCol("normalized_text") \
  .setOutputCol("document")

sentenceDetector = SentenceDetector()\
  .setInputCols(["document"])\
  .setOutputCol("sentence")

bertEmbeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased") \
  .setInputCols(["sentence"]) \
  .setOutputCol("bert_embeddings")

snomedTrainingPipeline = Pipeline(stages=[
  documentAssembler,
  sentenceDetector,
  bertEmbeddings
])
snomedTrainingModel = snomedTrainingPipeline.fit(data)
snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
bertExtractor = SentenceEntityResolverApproach() \
  .setNeighbours(25) \
  .setThreshold(1000) \
  .setInputCols(["bert_embeddings"]) \
  .setNormalizedCol("normalized_text") \
  .setLabelCol("label") \
  .setOutputCol("snomed_code") \
  .setDistanceFunction("EUCLIDIAN") \
  .setCaseSensitive(False)

snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture scala_example -%}
// Training a SNOMED resolution model using BERT sentence embeddings
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data and their labels.
val documentAssembler = new DocumentAssembler()
   .setInputCol("normalized_text")
   .setOutputCol("document")

val sentenceDetector = SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

 val bertEmbeddings = BertSentenceEmbeddings.pretrained("sent_biobert_pubmed_base_cased")
   .setInputCols("sentence")
   .setOutputCol("bert_embeddings")
 val snomedTrainingPipeline = new Pipeline().setStages(Array(
   documentAssembler,
   sentenceDetector,
   bertEmbeddings
 ))
 val snomedTrainingModel = snomedTrainingPipeline.fit(data)
 val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val bertExtractor = new SentenceEntityResolverApproach()
  .setNeighbours(25)
  .setThreshold(1000)
  .setInputCols("bert_embeddings")
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setOutputCol("snomed_code")
  .setDistanceFunction("EUCLIDIAN")
  .setCaseSensitive(false)

val snomedModel = bertExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture api_link -%}
[SentenceEntityResolverApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[SentenceEntityResolverApproach](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl.annotator.SentenceEntityResolverApproach.html)
{%- endcapture -%}


{% include templates/licensed_training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
%}
