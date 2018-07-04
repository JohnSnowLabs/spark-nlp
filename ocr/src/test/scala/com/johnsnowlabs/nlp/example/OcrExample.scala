package com.johnsnowlabs.example

import com.johnsnowlabs.nlp.{Annotation, OcrAssembler}
import com.johnsnowlabs.nlp.base.Finisher
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

import scala.collection.mutable

object OcrExample extends App {

  val spark = SparkSession.builder().master("local[4]").getOrCreate
  import spark.implicits._

  val ocrAnnotator = new OcrAssembler().
    setInputPath("./files"). // these are your files, possibly in HDFS
    //setPageSegmentationMode(3). // this causes different regions on each page be treated separately
    setOutputCol("text_regions") // this is where annotations end up being

  val finisher = new Finisher().setInputCols("text_regions")

  val pipeline = new Pipeline().setStages(Array(ocrAnnotator))

  // annotator doesn't use input DS, still it is used for accessing context, so it can't be null
  val empty = Seq.empty[String].toDS

  val result = pipeline.fit(empty).transform(empty).collect()
  val firstRow = result(0)(0).asInstanceOf[mutable.WrappedArray[Annotation]]
  val secondRow = result(1)(0).asInstanceOf[GenericRowWithSchema]

  result.foreach(println)

}
