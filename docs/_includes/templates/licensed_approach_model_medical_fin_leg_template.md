
<div class="tabs-box" markdown="1">

## {{include.title}}

{% include approachModelSelect.html %}

<div class="h3-box approach-content" markdown="1">

{{include.approach_description}}

**Input Annotator Types:** `{{include.approach_input_anno}}`

**Output Annotator Type:** `{{include.approach_output_anno}}`

{% if include.approach_note %}

> **Note:** {{include.approach_note}}

{% endif %}

{% if include.approach_source_link %}

| **Scala API:** {{include.approach_api_link}} | **Source:** {{include.approach_source_link}} |

{% else %}

| **Scala API:** {{include.approach_api_link}} |

{% endif %}


{% if include.approach_python_medical and include.approach_scala_medical %}

<details>

<summary class="button"><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-python active" markdown="1">

{% include programmingLanguageSelectScalaPythonMFL.html %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical active" markdown="1">

```python
{{include.approach_python_medical}}
```

</div>

<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```python
{{include.approach_python_finance}}
```

</div>

<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```python
{{include.approach_python_legal}}
```

</div>

</div>
<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-scala" markdown="1">

{% include programmingLanguageSelectScalaPythonMFL.html %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical active" markdown="1">

```scala
{{include.approach_scala_medical}}
```

</div>

<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```scala
{{include.approach_scala_finance}}
```

</div>

<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```scala
{{include.approach_scala_legal}}
```

</div>

</div>

</div>

</details>

{% endif %}

</div>

<div class="h3-box model-content" markdown="1" style="display: none;">

{{include.model_description}}

**Input Annotator Types:** `{{include.model_input_anno}}`

**Output Annotator Type:** `{{include.model_output_anno}}`

{% if include.model_note %}

> **Note:** {{include.model_note}}

{% endif %}

{% if include.model_source_link %}

| **Scala API:** {{include.model_api_link}} | **Source:** {{include.model_source_link}} |

{% else %}

| **Scala API:** {{include.model_api_link}} |

{% endif %}


{% if include.model_python_medical and include.model_scala_medical %}

<details>

<summary class="button"><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-python active" markdown="1">

{% include programmingLanguageSelectScalaPythonMFL.html %}


<div class="tabs-box-medic-inner-second highlighter-rouge language-medical active" markdown="1">

```python
{{include.model_python_medical}}
```

</div>
<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```python
{{include.model_python_finance}}
```

</div>
<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```python
{{include.model_python_legal}}
```

</div>

</div>
<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-scala" markdown="1">

{% include programmingLanguageSelectScalaPythonMFL.html %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical active" markdown="1">

```scala
{{include.model_scala_medical}}
```

</div>
<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```scala
{{include.model_scala_finance}}
```

</div>
<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```scala
{{include.model_scala_legal}}
```

</div>

</div>

</div>

</details>

{% endif %}

</div>

</div>
