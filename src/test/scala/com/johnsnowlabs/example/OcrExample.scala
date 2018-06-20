package com.johnsnowlabs.example

import com.johnsnowlabs.nlp.{OcrAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.base.Finisher
import org.apache.spark.ml.Pipeline

object OcrExample extends App {

  val spark = SparkAccessor.spark
  import spark.implicits._

  val ocrAnnotator = new OcrAssembler().
    setInputPath("./files"). // these are your files, possibly in HDFS
    //setPageSegmentationMode(3). // this causes different regions on each page be treated separately
    setOutputCol("text_regions") // this is where annotations end up being

  val finisher = new Finisher().setInputCols("text_regions")

  val pipeline = new Pipeline().setStages(Array(ocrAnnotator, finisher))

  // annotator doesn't use input DS, still it is used for accessing context, so it can't be null
  val empty = Seq.empty[String].toDS
  pipeline.fit(empty).transform(empty).show

}
