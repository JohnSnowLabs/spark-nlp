{%- capture title -%}
MistralTransformer
{%- endcapture -%}

{%- capture description -%}
[Mistral](https://huggingface.co/papers/2310.06825) is a 7B parameter language model, available as a pretrained and instruction-tuned variant, focused on balancing the scaling costs of large models with performance and efficient inference. This model uses sliding window attention (SWA) trained with a 8K context length and a fixed cache size to handle longer sequences more effectively. Grouped-query attention (GQA) speeds up inference and reduces memory requirements. Mistral also features a byte-fallback BPE tokenizer to improve token handling and efficiency by ensuring characters are never mapped to out-of-vocabulary tokens.

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val mistral = MistralTransformer.pretrained()
    .setMaxOutputLength(50)
    .setDoSample(False)
    .setInputCols(["document"])
    .setOutputCol("generation")
```
The default model is `"mistral_7b"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=MistralTransformer).

Spark NLP also supports importing Hugging Face Mistral models. See:
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:

- [Mistral 7B: Efficient open-weight language model (Paper)](https://arxiv.org/abs/2310.06825)  
- [Mistral AI on Hugging Face](https://huggingface.co/mistralai)  

**Paper abstract**

*We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license. *
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[MistralTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/MistralTransformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[MistralTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/mistral_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[MistralTransformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/MistralTransformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import MistralTransformer
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \\
    .setInputCol("text") \\
    .setOutputCol("document")

mistral = MistralTransformer.pretrained("mistral_7b", "en") \\
    .setInputCols(["document"]) \\
    .setOutputCol("embeddings")

pipeline = Pipeline().setStages([
    document_assembler,
    mistral
])

data = spark.createDataFrame([["Mistral models are efficient and powerful for NLP tasks."]], ["text"])

model = pipeline.fit(data)
results = model.transform(data)

results.selectExpr("explode(embeddings.embeddings) as vector").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.MistralTransformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val mistral = MistralTransformer.pretrained("mistral_7b", "en")
  .setInputCols("document")
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  mistral
))

val data = Seq("Mistral models are efficient and powerful for NLP tasks.").toDF("text")

val model = pipeline.fit(data)
val results = model.transform(data)

results.selectExpr("explode(embeddings.embeddings) as vector").show(false)
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
