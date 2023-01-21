
<div class="tabs-model-aproach" markdown="1">

## {{include.title}}

{% if include.approach and include.model %}

{% include approachModelSelect.html %}

{%- else -%}

<div class="tabs-model-aproach-head tac" markdown="1"></div>

{% endif %}

{% if include.approach %}

<!--Aproach-->
<div class="h3-box tabs-python-scala-box" markdown="1">

{{include.approach_description}}

**Input Annotator Types:** `{{include.approach_input_anno}}`

**Output Annotator Type:** `{{include.approach_output_anno}}`

{% if include.approach_note %}

> **Note:** {{include.approach_note}}

{% endif %}

{% if include.approach_python_api_link %}| **Python API:** {{include.approach_python_api_link}}{% endif %}{% if include.approach_api_link %}| **Scala API:** {{include.approach_api_link}}|{% endif %}

{% if include.approach_python_medical or include.approach_python_finance or include.approach_python_legal %}

<details>

<summary class="button"><b>Show Example</b></summary>

{% include programmingLanguageSelectScalaPython.html %}

<!--Python-->
<div class="tabs-mfl-box" markdown="1">

<div class="tabs-mfl-head" markdown="1">
{% if include.approach_python_medical %}<button class="tab-mfl-li" markdown="1">Medical</button>{% endif %}{% if include.approach_python_finance %}<button class="tab-mfl-li"  markdown="1">Finance</button>{% endif %}{% if include.approach_python_legal %}<button class="tab-mfl-li"  markdown="1">Legal</button>{% endif %}
</div>

{% if include.approach_python_medical %}

<div class="tab-mfl-content" markdown="1">

```python
{{include.approach_python_medical}}
```

</div>

{% endif %}
{% if include.approach_python_finance %}

<div class="tab-mfl-content" markdown="1">

```python
{{include.approach_python_finance}}
```

</div>

{% endif %}
{% if include.approach_python_legal %}

<div class="tab-mfl-content" markdown="1">

```python
{{include.approach_python_legal}}
```

</div>

{% endif %}

</div>
<!--END Python-->
<!--Scala-->
<div class="tabs-mfl-box" markdown="1">

<div class="tabs-mfl-head"  markdown="1">
{% if include.approach_scala_medical %}<button class="tab-mfl-li" markdown="1">Medical</button>{% endif %}{% if include.approach_scala_finance %}<button class="tab-mfl-li" markdown="1">Finance</button>{% endif %}{% if include.approach_scala_legal %}<button class="tab-mfl-li" markdown="1">Legal</button>{% endif %}
</div>

{% if include.approach_scala_medical %}

<div class="tab-mfl-content" markdown="1">

```scala
{{include.approach_scala_medical}}
```

</div>

{% endif %}
{% if include.approach_scala_finance %}

<div class="tab-mfl-content" markdown="1">

```scala
{{include.approach_scala_finance}}
```

</div>

{% endif %}
{% if include.approach_scala_legal %}

<div class="tab-mfl-content" markdown="1">

```scala
{{include.approach_scala_legal}}
```

</div>

{% endif %}

</div>
<!--END Scala-->

</details>
{% endif %}

</div>
<!--END Aproach-->

{% endif %}
{% if include.model %}

<!--Model-->
<div class="h3-box tabs-python-scala-box" markdown="1">

{{include.model_description}}

**Input Annotator Types:** `{{include.model_input_anno}}`

**Output Annotator Type:** `{{include.model_output_anno}}`

{% if include.model_note %}

> **Note:** {{include.model_note}}

{% endif %}


{% if include.model_python_api_link %}| **Python API:** {{include.model_python_api_link}}|{% endif %}{% if include.model_api_link %} **Scala API:** {{include.model_api_link}}|{% endif %}{% if include.model_source_link %} **Source:** {{include.model_source_link}}|{% endif %}


{% if include.model_python_medical or include.model_python_finance or include.model_python_legal %}

<details>

<summary class="button"><b>Show Example</b></summary>

{% include programmingLanguageSelectScalaPython.html %}

<!--Python-->
<div class="tabs-mfl-box" markdown="1">

<div class="tabs-mfl-head" markdown="1">
{% if include.model_python_medical %}<button class="tab-mfl-li" markdown="1">Medical</button>{% endif %}{% if include.model_python_finance %}<button class="tab-mfl-li" markdown="1">Finance</button>{% endif %}{% if include.model_python_legal %}<button class="tab-mfl-li" markdown="1">Legal</button>{% endif %}
</div>

{% if include.model_python_medical %}

<div class="tab-mfl-content" markdown="1">

```python
{{include.model_python_medical}}
```

</div>

{% endif %}
{% if include.model_python_finance %}

<div class="tab-mfl-content" markdown="1">

```python
{{include.model_python_finance}}
```

</div>

{% endif %}
{% if include.model_python_legal %}


<div class="tab-mfl-content" markdown="1">

```python
{{include.model_python_legal}}
```

</div>

{% endif %}

</div>
<!--END Python-->
<!--Scala--> 
<div class="tabs-mfl-box" markdown="1">

<div class="tabs-mfl-head" markdown="1">
{% if include.model_scala_medical %}<button class="tab-mfl-li"  markdown="1">Medical</button>{% endif %}{% if include.model_scala_finance %}<button class="tab-mfl-li"  markdown="1">Finance</button>{% endif %}{% if include.model_scala_finance %}<button class="tab-mfl-li"  markdown="1">Legal</button>{% endif %}
</div>

{% if include.model_scala_medical %}

<div class="tab-mfl-content" markdown="1">

```scala
{{include.model_scala_medical}}
```

</div>

{% endif %}
{% if include.model_scala_finance %}


<div class="tab-mfl-content" markdown="1">

```scala
{{include.model_scala_finance}}
```

</div>

{% endif %}
{% if include.model_scala_legal %}


<div class="tab-mfl-content" markdown="1">

```scala
{{include.model_scala_legal}}
```

</div>

{% endif %}

</div>
<!--END Scala--> 

</details>
{% endif %}

</div>
<!--END Model-->
{% endif %}

</div>