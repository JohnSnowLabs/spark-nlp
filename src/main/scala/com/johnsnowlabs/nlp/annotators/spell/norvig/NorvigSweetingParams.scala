package com.johnsnowlabs.nlp.annotators.spell.norvig

import org.apache.spark.ml.param.{BooleanParam, Params}

trait NorvigSweetingParams extends Params {


  val caseSensitive = new BooleanParam(this, "caseSensitive", "sensitivity on spell checking")
  val doubleVariants = new BooleanParam(this, "doubleVariants", "increase search at cost of performance")
  val shortCircuit = new BooleanParam(this, "shortCircuit", "increase performance at cost of accuracy")

  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  def setDoubleVariants(value: Boolean): this.type = set(doubleVariants, value)

  def setShortCircuit(value: Boolean): this.type = set(shortCircuit, value)

}
