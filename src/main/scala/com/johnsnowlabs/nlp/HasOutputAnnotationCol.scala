package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{Param, Params}

trait HasOutputAnnotationCol extends Params {

  protected final val outputCol: Param[String] =
    new Param(this, "outputCol", "the output annotation column")

  /** Overrides annotation column name when transforming */
  final def setOutputCol(value: String): this.type = set(outputCol, value)

  /** Gets annotation column name going to generate */
  final def getOutputCol: String = $(outputCol)

}
