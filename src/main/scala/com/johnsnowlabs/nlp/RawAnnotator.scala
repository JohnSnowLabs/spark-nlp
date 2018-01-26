package com.johnsnowlabs.nlp

import org.apache.spark.ml.Model

/**
  * Created by jose on 25/01/18.
  */
abstract class RawAnnotator[M<:Model[M]] extends Model[M]
    with ParamsAndFeaturesWritable
    with HasAnnotatorType
    with HasInputAnnotationCols
    with HasOutputAnnotationCol
    with HasWordEmbeddings {
}
