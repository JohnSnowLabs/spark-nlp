package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.BooleanParam

trait HasCaseSensitiveProperties extends ParamsAndFeaturesWritable {

  /** Whether to ignore case in index lookups (Default depends on model)
    *
    * @group param
    */
  val caseSensitive = new BooleanParam(this, "caseSensitive", "Whether to ignore case in index lookups")

  setDefault(caseSensitive, false)
  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** @group setParam */
  def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

}
