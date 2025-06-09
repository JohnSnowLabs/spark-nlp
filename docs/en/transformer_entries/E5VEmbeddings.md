{%- capture title -%}
E5VEmbeddings
{%- endcapture -%}

{%- capture description -%}
Universal multimodal embeddings using E5-V.

E5-V is a multimodal embedding model that bridges the modality gap between text and images, enabling strong performance in cross-modal retrieval, classification, clustering, and more. It supports both image+text and text-only embedding scenarios, and is fine-tuned from lmms-lab/llama3-llava-next-8b. The default model is `"e5v_1_5_7b_int4"`.

Note that this annotator is only supported for Spark Versions 3.4 and up.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = E5VEmbeddings.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("e5v")
```

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=E5V).

For extended examples of usage, see
[E5VEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/E5VEmbeddingsTestSpec.scala).

**Sources** :

- [E5-V: Universal Embeddings with Multimodal Large Language Models (arXiv)](https://arxiv.org/abs/2407.12580)
- [Hugging Face Model Card](https://huggingface.co/royokong/e5-v)
- [E5-V Github Repository](https://github.com/kongds/E5-V)
{%- endcapture -%}

{%- capture input_anno -%}
IMAGE
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
# Image + Text Embedding
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit

image_df = spark.read.format("image").option("dropInvalid", True).load(imageFolder)
imagePrompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\\nSummary above image in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
test_df = image_df.withColumn("text", lit(imagePrompt))
imageAssembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")
e5vEmbeddings = E5VEmbeddings.pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("e5v")
pipeline = Pipeline().setStages([
    imageAssembler,
    e5vEmbeddings
])
result = pipeline.fit(test_df).transform(test_df)
result.select("e5v.embeddings").show(truncate=False)

# Text-Only Embedding
from sparknlp.util import EmbeddingsDataFrameUtils
textPrompt = "<|start_header_id|>user<|end_header_id|>\n\n<sent>\\nSummary above sentence in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
textDesc = "A cat sitting in a box."
nullImageDF = spark.createDataFrame(
    spark.sparkContext.parallelize([EmbeddingsDataFrameUtils.emptyImageRow]),
    EmbeddingsDataFrameUtils.imageSchema)
textDF = nullImageDF.withColumn("text", lit(textPrompt.replace("<sent>", textDesc)))
e5vEmbeddings = E5VEmbeddings.pretrained() \
    .setInputCols(["image"]) \
    .setOutputCol("e5v")
result = e5vEmbeddings.transform(textDF)
result.select("e5v.embeddings").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
// Image + Text Embedding
import org.apache.spark.sql.functions.lit
import com.johnsnowlabs.nlp.base.ImageAssembler
import com.johnsnowlabs.nlp.embeddings.E5VEmbeddings
import org.apache.spark.ml.Pipeline

val imageDF = spark.read.format("image").option("dropInvalid", value = true).load(imageFolder)
val imagePrompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\\nSummary above image in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
val testDF = imageDF.withColumn("text", lit(imagePrompt))
val imageAssembler = new ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
val e5vEmbeddings = E5VEmbeddings.pretrained()
  .setInputCols("image_assembler")
  .setOutputCol("e5v")
val pipeline = new Pipeline().setStages(Array(imageAssembler, e5vEmbeddings))
val result = pipeline.fit(testDF).transform(testDF)
result.select("e5v.embeddings").show(truncate = false)

// Text-Only Embedding
import com.johnsnowlabs.nlp.util.EmbeddingsDataFrameUtils.{emptyImageRow, imageSchema}
val textPrompt = "<|start_header_id|>user<|end_header_id|>\n\n<sent>\\nSummary above sentence in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
val textDesc = "A cat sitting in a box."
val nullImageDF = spark.createDataFrame(spark.sparkContext.parallelize(Seq(emptyImageRow)), imageSchema)
val textDF = nullImageDF.withColumn("text", lit(textPrompt.replace("<sent>", textDesc)))
val e5vEmbeddings = E5VEmbeddings.pretrained()
  .setInputCols("image")
  .setOutputCol("e5v")
val result2 = e5vEmbeddings.transform(textDF)
result2.select("e5v.embeddings").show(truncate = false)
{%- endcapture -%}

{%- capture api_link -%}
[E5VEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/E5VEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[E5VEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/cv/e5v_embeddings/index.html#sparknlp.annotator.cv.e5v_embeddings.E5VEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[E5VEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/E5VEmbeddings.scala)
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