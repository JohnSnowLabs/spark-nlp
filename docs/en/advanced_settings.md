---
layout: docs
header: true
seotitle: Spark NLP - Advanced Settings
title: Spark NLP - Advanced Settings
permalink: /docs/en/advanced_settings
key: docs-install
modify_date: "2024-07-04"
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

</div><div class="h3-box" markdown="1">

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
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.0")
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
  --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.0
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
  --packages com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.0
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