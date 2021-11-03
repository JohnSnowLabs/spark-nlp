{%- capture title -%}
ChunkEntityResolver
{%- endcapture -%}

{%- capture model_description -%}
Returns a normalized entity for a particular trained ontology / curated dataset
(e.g. ICD-10, RxNorm, SNOMED etc).

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN, WORD_EMBEDDINGS
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
# Using pretrained models for SNOMED
# First the prior steps of the pipeline are defined.
# Output of types TOKEN and WORD_EMBEDDINGS are needed.
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
docAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("word_embeddings")
icdo_ner = MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "word_embeddings"]) \
    .setOutputCol("icdo_ner")
icdo_chunk = NerConverter().setInputCols(["sentence","token","icdo_ner"]).setOutputCol("icdo_chunk").setWhiteList(["Cancer"])
icdo_chunk_embeddings = ChunkEmbeddings() \
    .setInputCols(["icdo_chunk", "word_embeddings"]) \
    .setOutputCol("icdo_chunk_embeddings")
icdo_chunk_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical", "en", "clinical/models") \
    .setInputCols(["token","icdo_chunk_embeddings"]) \
    .setOutputCol("tm_icdo_code")
clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "word_embeddings"]) \
  .setOutputCol("ner")
ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")
ner_chunk_tokenizer = ChunkTokenizer() \
    .setInputCols(["ner_chunk"]) \
    .setOutputCol("ner_token")
ner_chunk_embeddings = ChunkEmbeddings() \
    .setInputCols(["ner_chunk", "word_embeddings"]) \
    .setOutputCol("ner_chunk_embeddings")

# Definition of the SNOMED Resolution
ner_snomed_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models") \
      .setInputCols(["ner_token","ner_chunk_embeddings"]).setOutputCol("snomed_result")
pipelineFull = Pipeline().setStages([
      docAssembler,
      sentenceDetector,
      tokenizer,
      word_embeddings,

      clinical_ner,
      ner_converter,
      ner_chunk_embeddings,
      ner_chunk_tokenizer,
      ner_snomed_resolver,

      icdo_ner,
      icdo_chunk,
      icdo_chunk_embeddings,
      icdo_chunk_resolver
])
pipelineModelFull = pipelineFull.fit(data)
result = pipelineModelFull.transform(data).cache()

# Show results
result.selectExpr("explode(snomed_result)")
  .selectExpr(
    "col.metadata.target_text",
    "col.metadata.resolved_text",
    "col.metadata.confidence",
    "col.metadata.all_k_results",
    "col.metadata.all_k_resolutions")
  .filter($"confidence" > 0.2).show(5)
+--------------------+--------------------+----------+--------------------+--------------------+
|         target_text|       resolved_text|confidence|       all_k_results|   all_k_resolutions|
+--------------------+--------------------+----------+--------------------+--------------------+
|hypercholesterolemia|Hypercholesterolemia|    0.2524|13644009:::267432...|Hypercholesterole...|
|                 CBC|             Neocyte|    0.4980|259680000:::11573...|Neocyte:::Blood g...|
|                CD38|       Hypoviscosity|    0.2560|47872005:::370970...|Hypoviscosity:::E...|
|           platelets| Increased platelets|    0.5267|6631009:::2596800...|Increased platele...|
|                CD38|       Hypoviscosity|    0.2560|47872005:::370970...|Hypoviscosity:::E...|
+--------------------+--------------------+----------+--------------------+--------------------+
{%- endcapture -%}

{%- capture model_scala_example -%}
// Using pretrained models for SNOMED
// First the prior steps of the pipeline are defined.
// Output of types TOKEN and WORD_EMBEDDINGS are needed.
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val docAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols("sentence", "token")
  .setOutputCol("word_embeddings")
val icdo_ner = MedicalNerModel.pretrained("ner_bionlp", "en", "clinical/models")
  .setInputCols("sentence", "token", "word_embeddings")
  .setOutputCol("icdo_ner")
val icdo_chunk = new NerConverter().setInputCols("sentence","token","icdo_ner").setOutputCol("icdo_chunk").setWhiteList("Cancer")
val icdo_chunk_embeddings = new ChunkEmbeddings()
  .setInputCols("icdo_chunk", "word_embeddings")
  .setOutputCol("icdo_chunk_embeddings")
val icdo_chunk_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_icdo_clinical", "en", "clinical/models")
  .setInputCols("token","icdo_chunk_embeddings")
  .setOutputCol("tm_icdo_code")
val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
.setInputCols("sentence", "token", "word_embeddings")
.setOutputCol("ner")
val ner_converter = new NerConverter()
.setInputCols("sentence", "token", "ner")
.setOutputCol("ner_chunk")
val ner_chunk_tokenizer = new ChunkTokenizer()
  .setInputCols("ner_chunk")
  .setOutputCol("ner_token")
val ner_chunk_embeddings = new ChunkEmbeddings()
  .setInputCols("ner_chunk", "word_embeddings")
  .setOutputCol("ner_chunk_embeddings")

// Definition of the SNOMED Resolution
val ner_snomed_resolver = ChunkEntityResolverModel.pretrained("chunkresolve_snomed_findings_clinical","en","clinical/models")
    .setInputCols("ner_token","ner_chunk_embeddings").setOutputCol("snomed_result")
val pipelineFull = new Pipeline().setStages(Array(
    docAssembler,
    sentenceDetector,
    tokenizer,
    word_embeddings,

    clinical_ner,
    ner_converter,
    ner_chunk_embeddings,
    ner_chunk_tokenizer,
    ner_snomed_resolver,

    icdo_ner,
    icdo_chunk,
    icdo_chunk_embeddings,
    icdo_chunk_resolver
))
val pipelineModelFull = pipelineFull.fit(data)
val result = pipelineModelFull.transform(data).cache()

// Show results
//
// result.selectExpr("explode(snomed_result)")
//   .selectExpr(
//     "col.metadata.target_text",
//     "col.metadata.resolved_text",
//     "col.metadata.confidence",
//     "col.metadata.all_k_results",
//     "col.metadata.all_k_resolutions")
//   .filter($"confidence" > 0.2).show(5)
// +--------------------+--------------------+----------+--------------------+--------------------+
// |         target_text|       resolved_text|confidence|       all_k_results|   all_k_resolutions|
// +--------------------+--------------------+----------+--------------------+--------------------+
// |hypercholesterolemia|Hypercholesterolemia|    0.2524|13644009:::267432...|Hypercholesterole...|
// |                 CBC|             Neocyte|    0.4980|259680000:::11573...|Neocyte:::Blood g...|
// |                CD38|       Hypoviscosity|    0.2560|47872005:::370970...|Hypoviscosity:::E...|
// |           platelets| Increased platelets|    0.5267|6631009:::2596800...|Increased platele...|
// |                CD38|       Hypoviscosity|    0.2560|47872005:::370970...|Hypoviscosity:::E...|
// +--------------------+--------------------+----------+--------------------+--------------------+
//
{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkEntityResolverModel](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverModel)
{%- endcapture -%}

{%- capture approach_description -%}
Contains all the parameters and methods to train a ChunkEntityResolverModel.
It transform a dataset with two Input Annotations of types TOKEN and WORD_EMBEDDINGS, coming from e.g. ChunkTokenizer
and ChunkEmbeddings Annotators and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please use ChunkEntityResolverModel
and see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN, WORD_EMBEDDINGS
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
# Training a SNOMED model
# Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data
# and their labels.
document = DocumentAssembler() \
    .setInputCol("normalized_text") \
    .setOutputCol("document")

chunk = Doc2Chunk() \
    .setInputCols(["document"]) \
    .setOutputCol("chunk")

token = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

chunkEmb = ChunkEmbeddings() \
        .setInputCols(["chunk", "embeddings"]) \
        .setOutputCol("chunk_embeddings")

snomedTrainingPipeline = Pipeline().setStages([
    document,
    chunk,
    token,
    embeddings,
    chunkEmb
])

snomedTrainingModel = snomedTrainingPipeline.fit(data)

snomedData = snomedTrainingModel.transform(data).cache()

# Then the Resolver can be trained with
snomedExtractor = ChunkEntityResolverApproach() \
    .setInputCols(["token", "chunk_embeddings"]) \
    .setOutputCol("recognized") \
    .setNeighbours(1000) \
    .setAlternatives(25) \
    .setNormalizedCol("normalized_text") \
    .setLabelCol("label") \
    .setEnableWmd(True).setEnableTfidf(True).setEnableJaccard(True) \
    .setEnableSorensenDice(True).setEnableJaroWinkler(True).setEnableLevenshtein(True) \
    .setDistanceWeights([1, 2, 2, 1, 1, 1]) \
    .setAllDistancesMetadata(True) \
    .setPoolingStrategy("MAX") \
    .setThreshold(1e32)
model = snomedExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_scala_example -%}
// Training a SNOMED model
// Define pre-processing pipeline for training data. It needs consists of columns for the normalized training data
// and their labels.
val document = new DocumentAssembler()
  .setInputCol("normalized_text")
  .setOutputCol("document")

val chunk = new Doc2Chunk()
  .setInputCols("document")
  .setOutputCol("chunk")

val token = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

val chunkEmb = new ChunkEmbeddings()
      .setInputCols("chunk", "embeddings")
      .setOutputCol("chunk_embeddings")

val snomedTrainingPipeline = new Pipeline().setStages(Array(
  document,
  chunk,
  token,
  embeddings,
  chunkEmb
))

val snomedTrainingModel = snomedTrainingPipeline.fit(data)

val snomedData = snomedTrainingModel.transform(data).cache()

// Then the Resolver can be trained with
val snomedExtractor = new ChunkEntityResolverApproach()
  .setInputCols("token", "chunk_embeddings")
  .setOutputCol("recognized")
  .setNeighbours(1000)
  .setAlternatives(25)
  .setNormalizedCol("normalized_text")
  .setLabelCol("label")
  .setEnableWmd(true).setEnableTfidf(true).setEnableJaccard(true)
  .setEnableSorensenDice(true).setEnableJaroWinkler(true).setEnableLevenshtein(true)
  .setDistanceWeights(Array(1, 2, 2, 1, 1, 1))
  .setAllDistancesMetadata(true)
  .setPoolingStrategy("MAX")
  .setThreshold(1e32)
val model = snomedExtractor.fit(snomedData)

{%- endcapture -%}

{%- capture approach_api_link -%}
[ChunkEntityResolverApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverApproach)
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
