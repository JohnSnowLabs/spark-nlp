
<div class="h3-box model-content" markdown="1">

### {{include.title}}

{{include.description}}

**Input Annotator Types:** `{{include.input_anno}}`

**Output Annotator Type:** `{{include.output_anno}}`

{% if include.note %}

> **Note:** {{include.note}}

{% endif %}

| **Python API:** {{include.python_api_link}} | **Scala API:** {{include.api_link}} |

{% if include.python_example and include.scala_example %}

<details>

<summary class="button"><b>Show Example</b></summary>

<div class="tabs-box tabs-new" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
{{include.python_example}}
```

```scala
{{include.scala_example}}
```

</div>

</details>

{% endif %}

</div>
