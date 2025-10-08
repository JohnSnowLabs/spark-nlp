package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, Param}

trait HasPdfReaderProperties extends ParamsAndFeaturesWritable {

  val titleThreshold: Param[Double] =
    new Param[Double](
      this,
      "titleThreshold",
      "Minimum font size threshold for title detection in PDF docs")

  def setTitleThreshold(value: Double): this.type = {
    set(titleThreshold, value)
  }

  final val readAsImage = new BooleanParam(this, "readAsImage", "Read PDF pages as images.")

  setDefault(titleThreshold -> 18, readAsImage -> false)

}
