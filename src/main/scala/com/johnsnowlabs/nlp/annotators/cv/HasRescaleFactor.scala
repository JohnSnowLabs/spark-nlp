package com.johnsnowlabs.nlp.annotators.cv

import org.apache.spark.ml.param.{BooleanParam, DoubleParam, Params}

/** Enables parameters to handle rescaling for image pre-processors. */
trait HasRescaleFactor {
  this: Params =>

  /** Whether to rescale the image values by rescaleFactor.
    *
    * @group param
    */
  val doRescale =
    new BooleanParam(this, "doRescale", "Whether to rescale the image values by rescaleFactor.")

  /** Factor to scale the image values (Default: `1 / 255.0`).
    *
    * @group param
    */
  val rescaleFactor =
    new DoubleParam(this, "rescaleFactor", "Factor to scale the image values")

  /** @group setParam */
  def setDoRescale(value: Boolean): this.type = set(this.doRescale, value)

  /** @group getParam */
  def getDoRescale: Boolean = $(doRescale)

  /** @group setParam */
  def setRescaleFactor(value: Double): this.type = set(this.rescaleFactor, value)

  /** @group getParam */
  def getRescaleFactor: Double = $(rescaleFactor)
}
