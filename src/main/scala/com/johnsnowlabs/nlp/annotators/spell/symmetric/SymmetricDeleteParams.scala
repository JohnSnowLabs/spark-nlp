package com.johnsnowlabs.nlp.annotators.spell.symmetric

import org.apache.spark.ml.param.{IntParam, Params}

trait SymmetricDeleteParams extends Params{

  val maxEditDistance = new IntParam(this, "maxEditDistance", "max edit distance characters to derive strings from a word")

  def setMaxEditDistance(value: Int): this.type = set(maxEditDistance, value)

}
