{%- capture title -%}
InstructorEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence embeddings using INSTRUCTOR.

Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text
embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation,
etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction,
without any finetuning. Instructor👨‍ achieves sota on 70 diverse embedding tasks!

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = InstructorEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("instructor_embeddings")
```

The default model is `"instructor_base"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=Instructor).

For extended examples of usage, see
[InstructorEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/InstructorEmbeddingsTestSpec.scala).

**Sources** :

[One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://arxiv.org/abs/2212.09741)

[INSTRUCTOR Github Repository](https://github.com/HKUNLP/instructor-embedding/)

**Paper abstract**

*We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions:
every text input is embedded together with instructions explaining the use case (e.g., task
and domain descriptions). Unlike encoders from prior work that are more specialized,
INSTRUCTOR is a single embedder that can generate text embeddings tailored to different
downstream tasks and domains, without any further training. We first annotate instructions for
330 diverse tasks and train INSTRUCTOR on this multitask mixture with a contrastive loss. We
evaluate INSTRUCTOR on 70 embedding evaluation tasks (66 of which are unseen during training),
ranging from classification and information retrieval to semantic textual similarity and text
generation evaluation. INSTRUCTOR, while having an order of magnitude fewer parameters than
the previous best model, achieves state-of-the-art performance, with an average improvement of
3.4% compared to the previous best results on the 70 diverse datasets. Our analysis suggests
that INSTRUCTOR is robust to changes in instructions, and that instruction finetuning
mitigates the challenge of training a single model on diverse datasets. Our model, code, and
data are available at this https URL. https://instructor-embedding.github.io/*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
embeddings = InstructorEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setInstruction("Represent the Medicine sentence for clustering: ") \
    .setOutputCol("instructor_embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["instructor_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.InstructorEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = InstructorEmbeddings.pretrained("instructor_base", "en")
  .setInputCols("document")
  .setInstruction("Represent the Medicine sentence for clustering: ")
  .setOutputCol("instructor_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("instructor_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  embeddingsFinisher
))

val data = Seq("Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[InstructorEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/InstructorEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[InstructorEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/instructor_embeddings/index.html#sparknlp.annotator.embeddings.instructor_embeddings.InstructorEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[InstructorEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/InstructorEmbeddings.scala)
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