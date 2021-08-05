---
layout: docs
header: true
title: Transformers
permalink: /docs/en/transformers
key: docs-transformers
modify_date: "2021-08-05"
use_language_switcher: "Python-Scala-Java"
---

<script> {% include scripts/approachModelSwitcher.js %} </script>

{% assign parent_path = "en/transformer_entries" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "transformer_entries/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}


## Import Transformers into Spark NLP

Please visit this discussion to learn how to import external Transformers models from HuggingFace and TF Hub into Spark NLP 🚀:

[Import Transformers into Spark NLP 🚀](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)