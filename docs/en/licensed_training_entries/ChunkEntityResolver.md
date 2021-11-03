{%- capture title -%}
ChunkEntityResolver
{%- endcapture -%}

{%- capture description -%}
Contains all the parameters and methods to train a ChunkEntityResolverModel.
It transform a dataset with two Input Annotations of types TOKEN and WORD_EMBEDDINGS, coming from e.g. ChunkTokenizer
and ChunkEmbeddings Annotators and returns the normalized entity for a particular trained ontology / curated dataset.
(e.g. ICD-10, RxNorm, SNOMED etc.)

To use pretrained models please use ChunkEntityResolverModel
and see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution) for available models.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN, WORD_EMBEDDINGS
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

{%- capture api_link -%}
[ChunkEntityResolverApproach](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/resolution/ChunkEntityResolverApproach)
{%- endcapture -%}


{%- capture python_api_link -%}
[ChunkEntityResolverApproach](https://nlp.johnsnowlabs.comlicensed/api/python/reference/autosummary/sparknlp_jsl.annotator.ChunkEntityResolverApproach.html)
{%- endcapture -%}


{% include templates/licensed_training_anno_template.md
title=title
model_description=model_description
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
%}
