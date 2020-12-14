---
layout: docs
header: true
title: Helper functions
permalink: /docs/en/auxiliary
key: docs-auxiliary
modify_date: "2019-11-28"
use_language_switcher: "Python-Scala"

---


<div class="h3-box" markdown="1">

### Spark NLP Annotation functions

The functions presented here help users manipulate annotations, by providing
both UDFs and dataframe utilities to deal with them more easily

</div><div class="h3-box" markdown="1">

#### Python
In python, the functions are straight forward and have both UDF and Dataframe applications
* `map_annotations(f, output_type: DataType)` UDF that applies f(). Requires output DataType from pyspark.sql.types
* `map_annotations_strict(f)` UDF that apples an f() method that returns a list of Annotations
* `map_annotations_col(dataframe: DataFrame, f, column, output_column, output_type)` applies f() to `column` from `dataframe` 
* `filter_by_annotations_col(dataframe, f, column)` applies a boolean filter f() to `column` from `dataframe`
* `explode_annotations_col(dataframe: DataFrame, column, output_column)` explodes annotation `column` from `dataframe`

</div><div class="h3-box" markdown="1">

#### Scala
In Scala, importing inner functions brings implicits that allow these functions to be applied directly on top of the dataframe
* `mapAnnotations(function: Seq[Annotation] => T, outputType: DataType)`
* `mapAnnotationsStrict(function: Seq[Annotation] => Seq[Annotation])`
* `mapAnnotationsCol[T: TypeTag](column: String, outputCol: String, function: Seq[Annotation] => T)`
* `eachAnnotationsCol[T: TypeTag](column: String, function: Seq[Annotation] => Unit)`
* `def explodeAnnotationsCol[T: TypeTag](column: String, outputCol: String)`

</div>

**Imports:**

<div class="tabs-box pt0" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from sparknlp.functions import *
from sparknlp.annotation import Annotation
```

```scala
import com.johnsnowlabs.nlp.functions._
import com.johnsnowlabs.nlp.Annotation
```

</div>

**Examples:**

Complete usage examples can be seen here:
https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/234-release-candidate/jupyter/annotation/english/spark-nlp-basics/spark-nlp-basics-functions.ipynb

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```scala
val modified = data.mapAnnotationsCol("pos", "mod_pos", (_: Seq[Annotation]) => {
      "hello world"
    })
```

```python
def my_annoation_map_function(annotations):
    return list(map(lambda a: Annotation(
        'my_own_type',
        a.begin,
        a.end,
        a.result,
        {'my_key': 'custom_annotation_data'},
        []), annotations))
        
result.select(
    map_annotations(my_annoation_map_function, Annotation.arrayType())('token')
).toDF("my output").show(truncate=False)
```

</div>