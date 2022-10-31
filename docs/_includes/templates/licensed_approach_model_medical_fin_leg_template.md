
<div class="tabs-box" markdown="1">

## {{include.title}}

{% if include.approach and include.model %}

{% include approachModelSelect.html %}

{% endif %}

{% if include.approach %}

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
{% if include.approach_python_medical or include.approach_python_finance or include.approach_python_legal %}

<details>

<summary class="button"><b>Show Example</b></summary>

<div class="tabs-box test-approach" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-python active" markdown="1">

<div class="top_tab_li toptab-second"  markdown="1">
{% if include.approach_python_medical %}<button data-type="medical" class="tab-li-inner"  markdown="1">Medical</button>{% endif %}{% if include.approach_python_finance %}<button data-type="finance" class="tab-li-inner"  markdown="1">Finance</button>{% endif %}{% if include.approach_python_legal %}<button data-type="legal" class="tab-li-inner"  markdown="1">Legal</button>{% endif %}
</div>

{% if include.approach_python_medical %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical" markdown="1">

```python
{{include.approach_python_medical}}
```

</div>

{% endif %}
{% if include.approach_python_finance %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```python
{{include.approach_python_finance}}
```

</div>

{% endif %}
{% if include.approach_python_legal %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```python
{{include.approach_python_legal}}
```

</div>

{% endif %}

</div>
<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-scala" markdown="1">

<div class="top_tab_li toptab-second"  markdown="1">
{% if include.approach_scala_medical %}<button data-type="medical" class="tab-li-inner"  markdown="1">Medical</button>{% endif %}{% if include.approach_scala_finance %}<button data-type="finance" class="tab-li-inner"  markdown="1">Finance</button>{% endif %}{% if include.approach_scala_legal %}<button data-type="legal" class="tab-li-inner"  markdown="1">Legal</button>{% endif %}
</div>

{% if include.approach_scala_medical %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical" markdown="1">

```scala
{{include.approach_scala_medical}}
```

</div>

{% endif %}
{% if include.approach_scala_finance %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```scala
{{include.approach_scala_finance}}
```

</div>

{% endif %}
{% if include.approach_scala_legal %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```scala
{{include.approach_scala_legal}}
```

</div>

{% endif %}

</div>

</div>

</details>
{% endif %}

</div>

{% endif %}
{% if include.model %}

<div class="h3-box model-content" markdown="1" {% if include.approach %} style="display: none;" {% endif %}>

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
{% if include.model_python_medical or include.model_python_finance or include.model_python_legal %}

<details>

<summary class="button"><b>Show Example</b></summary>

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-python active" markdown="1">

<div class="top_tab_li toptab-second"  markdown="1">
{% if include.model_python_medical %}<button data-type="medical" class="tab-li-inner"  markdown="1">Medical</button>{% endif %}{% if include.model_python_finance %}<button data-type="finance" class="tab-li-inner"  markdown="1">Finance</button>{% endif %}{% if include.model_python_legal %}<button data-type="legal" class="tab-li-inner"  markdown="1">Legal</button>{% endif %}
</div>

{% if include.model_python_medical %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical" markdown="1">

```python
{{include.model_python_medical}}
```

</div>

{% endif %}
{% if include.model_python_finance %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```python
{{include.model_python_finance}}
```

</div>

{% endif %}
{% if include.model_python_legal %}


<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```python
{{include.model_python_legal}}
```

</div>

{% endif %}

</div>
<div class="tabs-box-medic-inner tabs-wrapper highlighter-rouge language-scala" markdown="1">

<div class="top_tab_li toptab-second"  markdown="1">
{% if include.model_scala_medical %}<button data-type="medical" class="tab-li-inner"  markdown="1">Medical</button>{% endif %}{% if include.model_scala_finance %}<button data-type="finance" class="tab-li-inner"  markdown="1">Finance</button>{% endif %}{% if include.model_scala_finance %}<button data-type="legal" class="tab-li-inner"  markdown="1">Legal</button>{% endif %}
</div>

{% if include.model_scala_medical %}

<div class="tabs-box-medic-inner-second highlighter-rouge language-medical" markdown="1">

```scala
{{include.model_scala_medical}}
```

</div>

{% endif %}
{% if include.model_scala_finance %}


<div class="tabs-box-medic-inner-second highlighter-rouge language-finance" markdown="1">

```scala
{{include.model_scala_finance}}
```

</div>

{% endif %}
{% if include.model_scala_legal %}


<div class="tabs-box-medic-inner-second highlighter-rouge language-legal" markdown="1">

```scala
{{include.model_scala_legal}}
```

</div>

{% endif %}

</div>

</div>

</details>
{% endif %}

</div>
{% endif %}

</div>
