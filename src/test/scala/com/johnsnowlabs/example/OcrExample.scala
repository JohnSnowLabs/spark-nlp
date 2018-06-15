package com.johnsnowlabs.example

import com.johnsnowlabs.nlp.SparkAccessor
import ocr.tesseract.OcrAnnotator

object OcrExample extends App {

  val spark = SparkAccessor.spark
  import spark.implicits._

  val ocrAnnotator = new OcrAnnotator().
    setInputPath("./files"). // these are your files, possibly in HDFS
    setOutputCol("text_regions") // this is where annotations end up being

  // annotator doesn't use input DS, still it is used for accessing context, so it can't be null
  ocrAnnotator.transform(Seq.empty[String].toDS).show


}
