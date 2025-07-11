{%- capture title -%}
Phi4Transformer
{%- endcapture -%}

{%- capture description -%}
Text Generation using Microsoft Phi-4.

Phi4Transformer loads the state-of-the-art Phi-4 model for advanced reasoning, code generation, and general NLP tasks. Phi-4 is a 14B parameter, dense decoder-only Transformer model trained on 9.8T tokens, with a 16K context length, and is best suited for prompts in chat format. The model is designed for high-quality, advanced reasoning, math, code, and general NLP, with a focus on English (multilingual data ~8%).

**Key Features:**
- 14B parameters, dense decoder-only Transformer
- 16K context length
- Trained on 9.8T tokens (synthetic, public domain, academic, Q&A, code)
- Benchmarks: MMLU 84.8, HumanEval 82.6, GPQA 56.1, DROP 75.5, MATH 80.6
- Safety alignment via SFT and DPO, red-teamed by Microsoft AIRT
- Released under MIT License

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val phi4 = Phi4Transformer.pretrained()
     .setInputCols("document")
     .setOutputCol("generation")
```
The default model is `"phi-4"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://huggingface.co/microsoft/phi-4).

To see which models are compatible and how to import them see [Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

phi4 = Phi4Transformer.pretrained("phi-4") \
    .setInputCols(["documents"]) \
    .setMaxOutputLength(60) \
    .setOutputCol("generation")

pipeline = Pipeline().setStages([documentAssembler, phi4])

data = spark.createDataFrame([
    (1, "<|im_start|>system<|im_sep|>\nYou are a helpful assistant!\n<|im_start|>user<|im_sep|>\nWhat is Phi-4?\n<|im_start|>assistant<|im_sep|>\n")
]).toDF("id", "text")

result = pipeline.fit(data).transform(data)
result.select("generation.result").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val phi4 = Phi4Transformer.pretrained("phi-4")
  .setInputCols(Array("documents"))
  .setMaxOutputLength(60)
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, phi4))

val data = Seq(
  (1, "<|im_start|>system<|im_sep|>\nYou are a helpful assistant!\n<|im_start|>user<|im_sep|>\nWhat is Phi-4?\n<|im_start|>assistant<|im_sep|>\n")
).toDF("id", "text")

val result = pipeline.fit(data).transform(data)
result.select("generation.result").show(truncate = false)
{%- endcapture -%}

{%- capture api_link -%}
[Phi4Transformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/Phi4Transformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[Phi4Transformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/phi4_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[Phi4Transformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/Phi4Transformer.scala)
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