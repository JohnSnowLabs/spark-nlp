---
layout: docs
header: true
seotitle: Spark NLP - Migrating to Spark 4
title: Spark NLP - Migrating to Spark 4
permalink: /docs/en/spark4_migration
key: docs-install
modify_date: "2026-08-02"
show_nav: true
sidebar:
    nav: sparknlp
---

<div class="h3-box" markdown="1">

### Which jar to use

Spark NLP publishes two Maven artifacts for Apache Spark 4.x:

```bash
# Regular jar - Databricks (including Spark 4.0.0 / DBR 17.3+) and any Spark 4.0.1+
com.johnsnowlabs.nlp:spark-nlp_2.13:{{ site.sparknlp_version }}

# spark400 jar - plain, unpatched Spark 4.0.0 outside Databricks
com.johnsnowlabs.nlp:spark-nlp-spark400_2.13:{{ site.sparknlp_version }}
```

Databricks Runtime 17.3 identifies itself as Spark 4.0.0, but its internal build already includes a fix for [SPARK-52259](https://issues.apache.org/jira/browse/SPARK-52259), a `Param` class binary compatibility break present in real, unpatched Spark 4.0.0 and fixed upstream in 4.0.1. That's why Databricks needs the regular jar, built against the post-fix `Param` shape, instead of `spark400`, which matches real 4.0.0's pre-fix shape.

If you see `NoSuchMethodError` on `Param`'s constructor, you're using the wrong jar for your Spark 4.0.0 runtime.

</div>
