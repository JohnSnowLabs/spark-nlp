{%- capture title -%} StarCoderTransformer {%- endcapture -%} 
{%- capture description -%} 
StarCoder is a family of code generation models trained on permissively licensed code from GitHub, optimized for program synthesis, completion, and chat-based coding assistance. The models are based on the Fill-in-the-Middle (FIM) training objective, enabling them to generate code not just from left-to-right but also in the middle of existing code snippets.  

Pretrained models can be loaded with the `pretrained` method of the companion object:

```scala
val starcoder = StarCoderTransformer.pretrained("starcoder2_3b_int4", "en")
    .setInputCols("documents")
    .setOutputCol("generation")
```

The default model is `"starcoder2_3b_int4"`, if no name is provided. For available pretrained models please see the Models Hub. 

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=StarCoderTransformer).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [BigCode StarCoder announcement blog](https://huggingface.co/blog/starcoder)  
- [bigcode-project / starcoder (GitHub)](https://github.com/bigcode-project/starcoder)  
- [StarCoder on Hugging Face](https://huggingface.co/bigcode/starcoder)  

**Paper abstract**

*The BigCode community, an open-scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs), introduces StarCoder and StarCoderBase: 15.5B parameter models with 8K context length, infilling capabilities and fast large-batch inference enabled by multi-query attention. StarCoderBase is trained on 1 trillion tokens sourced from The Stack, a large collection of permissively licensed GitHub repositories with inspection tools and an opt-out process. We fine-tuned StarCoderBase on 35B Python tokens, resulting in the creation of StarCoder. We perform the most comprehensive evaluation of Code LLMs to date and show that StarCoderBase outperforms every open Code LLM that supports multiple programming languages and matches or outperforms the OpenAI code-cushman-001 model. Furthermore, StarCoder outperforms every model that is fine-tuned on Python, can be prompted to achieve 40\% pass@1 on HumanEval, and still retains its performance on other programming languages. We take several important steps towards a safe open-access model release, including an improved PII redaction pipeline and a novel attribution tracing tool, and make the StarCoder models publicly available under a more commercially viable version of the Open Responsible AI Model license.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[StarCoderTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/StarCoderTransformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[StarCoderTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/starcoder_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[StarCoderTransformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/StarCoderTransformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import StarCoderTransformer
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \ 
    .setInputCol("text") \ 
    .setOutputCol("document") \

starcoder = StarCoderTransformer.pretrained("starcoder2_3b_int4","en") \    
    .setInputCols("document") \     
    .setOutputCol("generation") \

pipeline = Pipeline().setStages([
    documentAssembler,
    starcoder
])

data = spark.createDataFrame([
    ["def fibonacci(n):"]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("generation.result").show(truncate=False)

+--------------------------------------------+
|result                                      |
+--------------------------------------------+
|[def fibonacci(n):\n    if n <= 1: return n]|
+--------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.StarCoderTransformer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val starcoder = StarCoderTransformer.pretrained("starcoder2_3b_int4", "en")
  .setInputCols("document")
  .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  starcoder
))

val data = spark.createDataFrame(Seq(
  Tuple1("def fibonacci(n):")
)).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("generation.result").show(false)

+--------------------------------------------+
|result                                      |
+--------------------------------------------+
|[def fibonacci(n):\n    if n <= 1: return n]|
+--------------------------------------------+
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
