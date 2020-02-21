package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.BooleanParam

trait HasCaseSensitiveProperties extends ParamsAndFeaturesWritable {

  val caseSensitive = new BooleanParam(this, "caseSensitive", "whether to ignore case in index lookups")

  setDefault(caseSensitive, false)
  def getCaseSensitive: Boolean = $(caseSensitive)
  def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

}
