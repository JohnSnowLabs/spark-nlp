{%- capture title -%}
LLMEntityExtractor
{%- endcapture -%}

{%- capture description -%}
LLMEntityExtractor is an annotator that performs entity extraction from text using Large
Language Models (LLMs) with structured JSON output via BNF grammars. It embeds AutoGGUFModel
directly and uses simple string matching to compute accurate character indices for extracted
entities.

This annotator follows the LangExtract pattern from Google Research, combining few-shot
prompting with constrained generation through llama.cpp BNF grammars to ensure valid JSON
output. The LLM generates responses in a structured format enforced by grammar:

```json
{
  "extractions": [
    {"entity": "MEDICATION", "text": "aspirin"},
    {"entity": "DOSAGE", "text": "250mg"}
  ]
}
```

The annotator then performs string matching to find the exact character positions of each entity
in the original text, outputting CHUNK annotations with accurate begin/end indices and chunk
indexing similar to other Spark NLP annotators.

Batch processing is used for performance, all documents are processed together in a single
LLM call via `multiComplete` for maximum throughput.

The model is instantiated directly and automatically loads the specified AutoGGUF model at
runtime. The default model is `"qwen3_4b_bf16_gguf"`.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models).

For extended examples of usage, see [llama_cpp_in_Spark_NLP_LLMEntityExtractor.ipynb](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama_cpp_in_Spark_NLP_LLMEntityExtractor.ipynb) and [LLMEntityExtractorTestSpec](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/dl/LLMEntityExtractorTestSpec.scala).

**Note**: To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
the number of GPU layers with the `setNGpuLayers` method.

When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
according to your hardware to avoid out-of-memory errors.

**Parameters:**

{:.table-model-big}
| Parameter | Description | Default |
|---|---|---|
| `pretrained` | Name of the AutoGGUF model to load for entity extraction | `"qwen3_4b_bf16_gguf"` |
| `entityTypes` | List of entity types to extract (used in prompt) | `["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME"]` |
| `promptTemplate` | Custom prompt template for entity extraction. Use `{entityTypes}` and `{examples}` placeholders | Built-in default prompt |
| `fewShotExamples` | Few-shot examples as array of `(input_text, json_output)` tuples to guide the model | Empty array |
| `caseSensitive` | Whether entity matching is case-sensitive | `false` |
| `nPredict` | Maximum number of tokens to predict | `500` |
| `nCtx` | Context size for the model | `4096` |
| `nGpuLayers` | Number of layers to offload to GPU | `99` |
| `temperature` | Sampling temperature | `0.1` |
| `batchSize` | Number of documents to process in parallel | `4` |
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
import pyspark.sql.functions as F
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

# Define few-shot examples to guide the model
medical_examples = [
    (
        "Patient takes aspirin 81mg daily.",
        '{"extractions": [{"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95}, {"entity": "DOSAGE", "text": "81mg", "confidence": 0.98}, {"entity": "FREQUENCY", "text": "daily", "confidence": 0.99}]}'
    )
]

entityExtractor = LLMEntityExtractor() \
    .pretrained("qwen3_4b_bf16_gguf") \
    .setInputCols(["document"]) \
    .setOutputCol("entities") \
    .setEntityTypes(["MEDICATION", "DOSAGE", "ROUTE", "FREQUENCY", "PERSON", "ORGANIZATION"]) \
    .setFewShotExamples(medical_examples) \
    .setNPredict(500) \
    .setNGpuLayers(99) \
    .setTemperature(0.1) \
    .setBatchSize(4)

pipeline = Pipeline().setStages([documentAssembler, entityExtractor])

data = spark.createDataFrame([
    ["John Smith visited the United Nations headquarters in New York on January 15th, 2024."],
    ["Dr. Emily Chen from Stanford University presented her research at the WHO conference in Geneva."],
    ["Apple Inc. CEO Tim Cook announced the new product line at their Cupertino campus on March 3rd."]
]).toDF("text")

result = pipeline.fit(data).transform(data)
result.select(
    F.explode("entities").alias("entity")
).select(
    F.col("entity.result").alias("text"),
    F.col("entity.begin").alias("begin"),
    F.col("entity.end").alias("end"),
    F.col("entity.metadata.entity").alias("entity_type"),
    F.col("entity.metadata.chunk").alias("chunk_index")
).show(truncate=False)

+-------------------+-----+---+------------+-----------+
|text               |begin|end|entity_type |chunk_index|
+-------------------+-----+---+------------+-----------+
|John Smith         |0    |9  |PERSON      |0          |
|United Nations     |23   |36 |ORGANIZATION|1          |
|New York           |54   |61 |LOCATION    |2          |
|January 15th, 2024 |66   |83 |DATE        |3          |
|Dr. Emily Chen     |0    |13 |PERSON      |0          |
|Stanford University|20   |38 |ORGANIZATION|1          |
|Geneva             |88   |93 |LOCATION    |2          |
|WHO conference     |70   |83 |DATE        |3          |
|Apple Inc.         |0    |9  |ORGANIZATION|0          |
|Tim Cook           |15   |22 |PERSON      |1          |
|Cupertino          |64   |72 |LOCATION    |2          |
|March 3rd          |84   |92 |DATE        |3          |
+-------------------+-----+---+------------+-----------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import org.apache.spark.sql.functions._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators.ner.dl.LLMEntityExtractor
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

// Define few-shot examples to guide the model
val medicalExamples = Array(
  (
    "Patient takes aspirin 81mg daily.",
    """{"extractions": [{"entity": "MEDICATION", "text": "aspirin", "confidence": 0.95}, {"entity": "DOSAGE", "text": "81mg", "confidence": 0.98}, {"entity": "FREQUENCY", "text": "daily", "confidence": 0.99}]}"""
  )
)

val entityExtractor = new LLMEntityExtractor()
  .pretrained("qwen3_4b_bf16_gguf")
  .setInputCols("document")
  .setOutputCol("entities")
  .setEntityTypes(Array("MEDICATION", "DOSAGE", "ROUTE", "FREQUENCY", "PERSON", "ORGANIZATION"))
  .setFewShotExamples(medicalExamples)
  .setNPredict(500)
  .setNGpuLayers(99)
  .setTemperature(0.1f)
  .setBatchSize(4)

val pipeline = new Pipeline().setStages(Array(documentAssembler, entityExtractor))

val data = Seq(
  "John Smith visited the United Nations headquarters in New York on January 15th, 2024.",
  "Dr. Emily Chen from Stanford University presented her research at the WHO conference in Geneva.",
  "Apple Inc. CEO Tim Cook announced the new product line at their Cupertino campus on March 3rd."
).toDF("text")

val result = pipeline.fit(data).transform(data)
result.select(
    F.explode("entities").alias("entity")
).select(
    F.col("entity.result").alias("text"),
    F.col("entity.begin").alias("begin"),
    F.col("entity.end").alias("end"),
    F.col("entity.metadata.entity").alias("entity_type"),
    F.col("entity.metadata.chunk").alias("chunk_index")
).show(truncate=False)

+-------------------+-----+---+------------+-----------+
|text               |begin|end|entity_type |chunk_index|
+-------------------+-----+---+------------+-----------+
|John Smith         |0    |9  |PERSON      |0          |
|United Nations     |23   |36 |ORGANIZATION|1          |
|New York           |54   |61 |LOCATION    |2          |
|January 15th, 2024 |66   |83 |DATE        |3          |
|Dr. Emily Chen     |0    |13 |PERSON      |0          |
|Stanford University|20   |38 |ORGANIZATION|1          |
|Geneva             |88   |93 |LOCATION    |2          |
|WHO conference     |70   |83 |DATE        |3          |
|Apple Inc.         |0    |9  |ORGANIZATION|0          |
|Tim Cook           |15   |22 |PERSON      |1          |
|Cupertino          |64   |72 |LOCATION    |2          |
|March 3rd          |84   |92 |DATE        |3          |
+-------------------+-----+---+------------+-----------+
{%- endcapture -%}

{%- capture api_link -%}
[LLMEntityExtractor](/api/com/johnsnowlabs/nlp/annotators/ner/dl/LLMEntityExtractor)
{%- endcapture -%}

{%- capture python_api_link -%}
[LLMEntityExtractor](/api/python/reference/autosummary/sparknlp/annotator/ner/llm_entity_extractor/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[LLMEntityExtractor](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ner/dl/LLMEntityExtractor.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}