{%- capture title -%}
SentenceEntityResolver
{%- endcapture -%}

{%- capture model_description -%}
The model transforms a dataset with Input Annotation type SENTENCE_EMBEDDINGS, coming from e.g.
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings)
and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture model_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture model_output_anno -%}
ENTITY
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Resolving CPT
# First define pipeline stages to extract entities
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
sentenceDetector = SentenceDetectorDLModel.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk") \
    .setWhiteList(["Test","Procedure"])
c2doc = Chunk2Doc() \
    .setInputCols(["ner_chunk"]) \
    .setOutputCol("ner_chunk_doc")
sbert_embedder = BertSentenceEmbeddings \
    .pretrained("sbiobert_base_cased_mli","en","clinical/models") \
    .setInputCols(["ner_chunk_doc"]) \
    .setOutputCol("sbert_embeddings")

# Then the resolver is defined on the extracted entities and sentence embeddings
cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_augmented","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("cpt_code") \
    .setDistanceFunction("EUCLIDEAN")
sbert_pipeline_cpt = Pipeline().setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    c2doc,
    sbert_embedder,
    cpt_resolver])

sbert_outputs = sbert_pipeline_cpt.fit(data_ner).transform(data)
# Show results
#
# sbert_outputs
#   .select("explode(arrays_zip(ner_chunk.result ,ner_chunk.metadata, cpt_code.result, cpt_code.metadata, ner_chunk.begin, ner_chunk.end)) as cpt_code")
#   .selectExpr(
#     "cpt_code['0'] as chunk",
#     "cpt_code['1'].entity as entity",
#     "cpt_code['2'] as code",
#     "cpt_code['3'].confidence as confidence",
#     "cpt_code['3'].all_k_resolutions as all_k_resolutions",
#     "cpt_code['3'].all_k_results as all_k_results"
#   ).show(5)
# +--------------------+---------+-----+----------+--------------------+--------------------+
# |               chunk|   entity| code|confidence|   all_k_resolutions|         all_k_codes|
# +--------------------+---------+-----+----------+--------------------+--------------------+
# |          heart cath|Procedure|93566|    0.1180|CCA - Cardiac cat...|93566:::62319:::9...|
# |selective coronar...|     Test|93460|    0.1000|Coronary angiogra...|93460:::93458:::9...|
# |common femoral an...|     Test|35884|    0.1808|Femoral artery by...|35884:::35883:::3...|
# |   StarClose closure|Procedure|33305|    0.1197|Heart closure:::H...|33305:::33300:::3...|
# |         stress test|     Test|93351|    0.2795|Cardiovascular st...|93351:::94621:::9...|
# +--------------------+---------+-----+----------+--------------------+--------------------+
#
{%- endcapture -%}

{%- capture model_scala_example -%}
// Resolving CPT
// First define pipeline stages to extract entities
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")
val sentenceDetector = SentenceDetectorDLModel.pretrained()
  .setInputCols("document")
  .setOutputCol("sentence")
val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")
val clinical_ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")
val ner_converter = new NerConverter()
  .setInputCols("sentence", "token", "ner")
  .setOutputCol("ner_chunk")
  .setWhiteList("Test","Procedure")
val c2doc = new Chunk2Doc()
  .setInputCols("ner_chunk")
  .setOutputCol("ner_chunk_doc")
val sbert_embedder = BertSentenceEmbeddings
  .pretrained("sbiobert_base_cased_mli","en","clinical/models")
  .setInputCols("ner_chunk_doc")
  .setOutputCol("sbert_embeddings")

// Then the resolver is defined on the extracted entities and sentence embeddings
val cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_augmented","en", "clinical/models")
  .setInputCols("ner_chunk", "sbert_embeddings")
  .setOutputCol("cpt_code")
  .setDistanceFunction("EUCLIDEAN")
val sbert_pipeline_cpt = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tokenizer,
  word_embeddings,
  clinical_ner,
  ner_converter,
  c2doc,
  sbert_embedder,
  cpt_resolver))

// Show results
//
// sbert_outputs
//   .select("explode(arrays_zip(ner_chunk.result ,ner_chunk.metadata, cpt_code.result, cpt_code.metadata, ner_chunk.begin, ner_chunk.end)) as cpt_code")
//   .selectExpr(
//     "cpt_code['0'] as chunk",
//     "cpt_code['1'].entity as entity",
//     "cpt_code['2'] as code",
//     "cpt_code['3'].confidence as confidence",
//     "cpt_code['3'].all_k_resolutions as all_k_resolutions",
//     "cpt_code['3'].all_k_results as all_k_results"
//   ).show(5)
// +--------------------+---------+-----+----------+--------------------+--------------------+
// |               chunk|   entity| code|confidence|   all_k_resolutions|         all_k_codes|
// +--------------------+---------+-----+----------+--------------------+--------------------+
// |          heart cath|Procedure|93566|    0.1180|CCA - Cardiac cat...|93566:::62319:::9...|
// |selective coronar...|     Test|93460|    0.1000|Coronary angiogra...|93460:::93458:::9...|
// |common femoral an...|     Test|35884|    0.1808|Femoral artery by...|35884:::35883:::3...|
// |   StarClose closure|Procedure|33305|    0.1197|Heart closure:::H...|33305:::33300:::3...|
// |         stress test|     Test|93351|    0.2795|Cardiovascular st...|93351:::94621:::9...|
// +--------------------+---------+-----+----------+--------------------+--------------------+
//
{%- endcapture -%}

{%- capture model_api_link -%}
[SentenceEntityResolverModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverModel)
{%- endcapture -%}

{%- capture approach_description -%}
Contains all the parameters and methods to train a SentenceEntityResolverModel.
The model transforms a dataset with Input Annotation type SENTENCE_EMBEDDINGS, coming from e.g.
[BertSentenceEmbeddings](/docs/en/transformers#bertsentenceembeddings)
and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please use SentenceEntityResolverModel
and see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture approach_input_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_output_anno -%}
ENTITY
{%- endcapture -%}

{%- capture approach_python_example -%}
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

{%- capture approach_scala_example -%}
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

{%- capture approach_api_link -%}
[SentenceEntityResolverApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/SentenceEntityResolverApproach)
{%- endcapture -%}


{% include templates/licensed_approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_api_link=model_api_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
%}
