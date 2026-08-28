---
layout: docs
header: true
seotitle: Spark NLP - Advanced Settings
title: Spark NLP - Advanced Settings
permalink: /docs/en/advanced_settings
key: docs-install
modify_date: "2026-08-27"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

## SparkNLP Properties

You can change the following Spark NLP configurations via Spark Configuration:

{:.table-model-big}
| Property Name                                           | Default              | Meaning                                                                                                                                                                                                                                                                            |
|---------------------------------------------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `spark.jsl.settings.pretrained.cache_folder`            | `~/cache_pretrained` | The location to download and extract pretrained `Models` and `Pipelines`. By default, it will be in User's Home directory under `cache_pretrained` directory                                                                                                                       |
| `spark.jsl.settings.storage.cluster_tmp_dir`            | `hadoop.tmp.dir`     | The location to use on a cluster for temporarily files such as unpacking indexes for WordEmbeddings. By default, this locations is the location of `hadoop.tmp.dir` set via Hadoop configuration for Apache Spark. NOTE: `S3` is not supported and it must be local, HDFS, or DBFS |
| `spark.jsl.settings.annotator.log_folder`               | `~/annotator_logs`   | The location to save logs from annotators during training such as `NerDLApproach`, `ClassifierDLApproach`, `SentimentDLApproach`, `MultiClassifierDLApproach`, etc. By default, it will be in User's Home directory under `annotator_logs` directory                               |
| `spark.jsl.settings.aws.credentials.access_key_id`      | `None`               | Your AWS access key to use your S3 bucket to store log files of training models or access tensorflow graphs used in `NerDLApproach`                                                                                                                                                |
| `spark.jsl.settings.aws.credentials.secret_access_key`  | `None`               | Your AWS secret access key to use your S3 bucket to store log files of training models or access tensorflow graphs used in `NerDLApproach`                                                                                                                                         |
| `spark.jsl.settings.aws.credentials.session_token`      | `None`               | Your AWS MFA session token to use your S3 bucket to store log files of training models or access tensorflow graphs used in `NerDLApproach`                                                                                                                                         |
| `spark.jsl.settings.aws.s3_bucket`                      | `None`               | Your AWS S3 bucket to store log files of training models or access tensorflow graphs used in `NerDLApproach`                                                                                                                                                                       |
| `spark.jsl.settings.aws.region`                         | `None`               | Your AWS region to use your S3 bucket to store log files of training models or access tensorflow graphs used in `NerDLApproach`                                                                                                                                                    |
| `spark.jsl.settings.onnx.gpuDeviceId`                   | `0`                  | Constructs CUDA execution provider options for the specified non-negative device id.                                                                                                                                                                                               |
| `spark.jsl.settings.onnx.intraOpNumThreads`             | `6`                  | Sets the size of the CPU thread pool used for executing a single graph, if executing on a CPU.                                                                                                                                                                                     |
| `spark.jsl.settings.onnx.optimizationLevel`             | `ALL_OPT`            | Sets the optimization level of this options object, overriding the old setting.                                                                                                                                                                                                    |
| `spark.jsl.settings.onnx.executionMode`                 | `SEQUENTIAL`         | Sets the execution mode of this options object, overriding the old setting.                                                                                                                                                                                                        |
| `spark.jsl.settings.onnx.cuda.preload.mode`              | `search`             | Controls failure-first ONNX CUDA native-dependency recovery: `off`, `search`, or `explicit`. Recovery runs only after a recognized CUDA-provider dependency-loading failure.                                                                                                       |
| `spark.jsl.settings.onnx.cuda.preload.paths`             | Empty                | In `search` mode, a platform path-separated list of trusted directories with highest precedence. In `explicit` mode, the complete ordered list of absolute CUDA library files required by the packaged ONNX Runtime provider.                                                       |
| `spark.jsl.settings.serialization.fallbackLogMode`      | `off`                | Controls model fallback-loader diagnostics. Valid values are `off`, `summary`, and `full` (case-insensitive). This setting changes observability only; it does not enable or disable fallback loading.                                                                              |

### Fallback loader logging

When primary model deserialization fails but a compatibility fallback is available, use
`spark.jsl.settings.serialization.fallbackLogMode` to control the model-level diagnostic:

- `off` (default): do not emit a fallback-loader message;
- `summary`: emit one single-line warning with the model type and root cause; dynamic exception
  text is limited to 200 characters;
- `full`: emit the same summary and the complete original exception stack trace.

Unsupported values are treated as `off` and produce a concise configuration warning. The mode
controls logging only: fallback execution and error propagation remain unchanged. It suppresses
only the Spark NLP model-level fallback message, so Spark may still emit executor errors before
the fallback reader begins.

**Scala:**

```scala
spark.conf.set(
  "spark.jsl.settings.serialization.fallbackLogMode",
  "summary")
```

**Python:**

```python
spark.conf.set(
    "spark.jsl.settings.serialization.fallbackLogMode",
    "summary",
)
```

**spark-submit:**

```bash
spark-submit \
  --conf spark.jsl.settings.serialization.fallbackLogMode=full \
  your_application.py
```

</div><div class="h3-box" markdown="1">

### ONNX CUDA native-dependency recovery

Spark NLP first registers the ONNX Runtime CUDA provider normally. If registration succeeds, it does not inspect preload configuration, search the filesystem, call `System.load`, or retry. If registration fails because a required CUDA shared library cannot be loaded, Spark NLP can preload already-installed libraries inside the executor JVM and retry provider registration once with fresh provider and session-option objects.

- `search` (default): resolve the packaged provider's dependency manifest from trusted operator directories, runtime and linker paths, and a bounded search under generic installation roots. Resolution fails on missing or ambiguous candidates, and every group- or world-writable library is rejected.
- `explicit`: validate the configured complete ordered list of absolute regular files and load those exact canonical files without searching. A group-writable library is accepted only when its owner and group exactly match the authenticated executor identity; use this exception only on a platform with a documented guarantee that the executor group is exclusive to the authenticated executor identity.
- `off`: preserve the legacy failure path without reading the paths setting, discovery, preload, or retry.

All enabled modes fail closed. Recovery never silently falls back to CPU, does not install or bundle CUDA, does not use Python discovery, and is independent from model warm-up. Native-library paths are runtime Spark configuration and are not persisted in models.

Before any native load, Spark NLP canonicalizes and validates the complete manifest. Each library must be a readable ELF64 shared object whose architecture and uniquely mapped `DT_SONAME` match the current runtime and required manifest entry. The library and every canonical ancestor must be owned by `root` or the authenticated POSIX executor identity. Library files must not be world-writable. Search mode rejects group-writable libraries; explicit mode permits group-write only for a file whose owner and group exactly match the authenticated executor identity. Every canonical ancestor remains non-group-writable and non-world-writable. Filesystems without POSIX ownership and permission attributes are rejected. Search symlinks must remain inside their approved canonical root, and directory symlinks are not traversed.

Filesystem work is capped across discovery tiers at 20,000 visited entries, including path-list entries and configured directory canonicalization attempts. The configured preload path-list value and the aggregate runtime-derived path-list text each have an independent 1 MiB cap before trimming or splitting. Generic installation-root and linker-configuration traversal has a maximum depth of 6. The aggregate linker-configuration input across the root file and all recursive includes is capped at 1 MiB. Fixed-path `ldconfig -p` output is separately capped at 1 MiB, and `ldconfig` must terminate within 2 seconds after normal completion or forced destruction. Exceeding any limit fails recovery instead of broadening the search.

Example explicit configuration on Linux:

```bash
--conf spark.jsl.settings.onnx.cuda.preload.mode=explicit \
--conf spark.jsl.settings.onnx.cuda.preload.paths=/absolute/cuda/libcudart.so.12:/absolute/cuda/libcublasLt.so.12:/absolute/cuda/libcublas.so.12:/absolute/cuda/libcurand.so.10:/absolute/cuda/libcufft.so.11:/absolute/cuda/libcudnn.so.9
```

The exact SONAME inventory is version-coupled to the ONNX Runtime GPU provider packaged by Spark NLP. Reconcile it against `libonnxruntime_providers_cuda.so` whenever the ONNX Runtime dependency changes.

### How to set Spark NLP Configuration

**SparkSession:**

You can use `.config()` during SparkSession creation to set Spark NLP configurations.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder
    .master("local[*]")
    .config("spark.driver.memory", "16G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max", "2000m")
    .config("spark.jsl.settings.pretrained.cache_folder", "sample_data/pretrained")
    .config("spark.jsl.settings.storage.cluster_tmp_dir", "sample_data/storage")
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.13:{{ site.sparknlp_version }}")
    .getOrCreate()
```

**spark-shell:**

```sh
spark-shell \
  --driver-memory 16g \
  --conf spark.driver.maxResultSize=0 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
  --conf spark.kryoserializer.buffer.max=2000M \
  --conf spark.jsl.settings.pretrained.cache_folder="sample_data/pretrained" \
  --conf spark.jsl.settings.storage.cluster_tmp_dir="sample_data/storage" \
  --packages com.johnsnowlabs.nlp:spark-nlp_2.13:{{ site.sparknlp_version }}
```

**pyspark:**

```sh
pyspark \
  --driver-memory 16g \
  --conf spark.driver.maxResultSize=0 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer
  --conf spark.kryoserializer.buffer.max=2000M \
  --conf spark.jsl.settings.pretrained.cache_folder="sample_data/pretrained" \
  --conf spark.jsl.settings.storage.cluster_tmp_dir="sample_data/storage" \
  --packages com.johnsnowlabs.nlp:spark-nlp_2.13:{{ site.sparknlp_version }}
```

**Databricks:**

On a new cluster or existing one you need to add the following to the `Advanced Options -> Spark` tab:

```bash
spark.kryoserializer.buffer.max 2000M
spark.serializer org.apache.spark.serializer.KryoSerializer
spark.jsl.settings.pretrained.cache_folder dbfs:/PATH_TO_CACHE
spark.jsl.settings.storage.cluster_tmp_dir dbfs:/PATH_TO_STORAGE
spark.jsl.settings.annotator.log_folder dbfs:/PATH_TO_LOGS
```

NOTE: If this is an existing cluster, after adding new configs or changing existing properties you need to restart it.

#### Additional Configuration for Databricks
When running Email Reader feature `sparknlp.read().email("./email-files")` on Databricks, it is necessary to include the following Spark configurations to avoid dependency conflicts:

```bash
spark.driver.userClassPathFirst true
spark.executor.userClassPathFirst true
```
These configurations are required because the Databricks runtime environment includes a bundled version of the `com.sun.mail:jakarta.mail` library, which conflicts with `jakarta.activation`.
By setting these properties, the application ensures that the user-provided libraries take precedence over those bundled in the Databricks environment, resolving the dependency conflict.

#### Databricks Unity Catalog Volumes and pretrained models

Databricks documents that some JVM-based operations do not support reading from or writing to Unity Catalog Volumes through standard `/Volumes/...` paths. See the official Databricks guidance here:

[Databricks documentation: Work with files on Databricks](https://docs.databricks.com/aws/en/files/)

Spark NLP pretrained downloads rely on JVM-side file operations for download, move, and unzip. Because of this Databricks limitation, Unity Catalog Volumes are not supported as Spark NLP download/cache targets for `spark.jsl.settings.pretrained.cache_folder`, `spark.jsl.settings.storage.cluster_tmp_dir`, or `spark.jsl.settings.annotator.log_folder`.

For Databricks environments that store pretrained models on a Unity Catalog Volume, the supported workaround is to place the model artifacts on the Volume outside the Spark NLP `.pretrained()` flow and then load them directly with `.load(model_path)`.

**Load a model already stored on a Unity Catalog Volume**

```python
from sparknlp.annotator import NerDLModel

model_path = "/Volumes/<catalog>/<schema>/<volume>/cache_pretrained/ner_dl_en_2.4.3_2.4_1584624950746"

ner_model = NerDLModel.load(model_path) \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
```

</div><div class="h3-box" markdown="1">

### S3 Integration

**Logging:**

To configure S3 path for logging while training models. We need to set up AWS credentials as well as an S3 path

```bash
spark.conf.set("spark.jsl.settings.annotator.log_folder", "s3://my/s3/path/logs")
spark.conf.set("spark.jsl.settings.aws.credentials.access_key_id", "MY_KEY_ID")
spark.conf.set("spark.jsl.settings.aws.credentials.secret_access_key", "MY_SECRET_ACCESS_KEY")
spark.conf.set("spark.jsl.settings.aws.s3_bucket", "my.bucket")
spark.conf.set("spark.jsl.settings.aws.region", "my-region")
```

Now you can check the log on your S3 path defined in *spark.jsl.settings.annotator.log_folder* property.
Make sure to use the prefix *s3://*, otherwise it will use the default configuration.

**Tensorflow Graphs:**

To reference S3 location for downloading graphs. We need to set up AWS credentials

```bash
spark.conf.set("spark.jsl.settings.aws.credentials.access_key_id", "MY_KEY_ID")
spark.conf.set("spark.jsl.settings.aws.credentials.secret_access_key", "MY_SECRET_ACCESS_KEY")
spark.conf.set("spark.jsl.settings.aws.region", "my-region")
```

**MFA Configuration:**

In case your AWS account is configured with MFA. You will need first to get temporal credentials and add session token
to the configuration as shown in the examples below
For logging:

```bash
spark.conf.set("spark.jsl.settings.aws.credentials.session_token", "MY_TOKEN")
```

An example of a bash script that gets temporal AWS credentials can be
found [here](https://github.com/JohnSnowLabs/spark-nlp/blob/master/scripts/aws_tmp_credentials.sh)
This script requires three arguments:

```bash
./aws_tmp_credentials.sh iam_user duration serial_number
```

</div>
