package com.johnsnowlabs.storage

import org.apache.spark.ml.param.{BooleanParam, Params}

trait HasExcludableStorage extends Params {

  val includeStorage: BooleanParam = new BooleanParam(this, "includeStorage", "whether to include indexed storage in trained model")

  def setIncludeStorage(value: Boolean): this.type = set(includeStorage, value)

  def getIncludeStorage: Boolean = $(includeStorage)

  setDefault(includeStorage, true)

}
