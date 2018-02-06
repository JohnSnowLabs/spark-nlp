package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.nlp.annotators.param.WritableAnnotatorComponent
import com.johnsnowlabs.nlp.serialization.SerializedExternalResource

/** This represents an external source which contains information into
  * how an external resource shall be read by Spark-NLP's Resource Helper */
case class ExternalResource(
                             path: String,
                             readAs: ReadAs.Format,
                             options: Map[String, String]
                           ) extends WritableAnnotatorComponent {

  if (readAs == ReadAs.SPARK_DATASET)
    require(options.contains("format"), "Created ExternalResource to read as SPARK_DATASET but key 'format' " +
      "in options is not provided. Can be any spark.read.format type. e.g. 'text' or 'json' or 'parquet'")

  override def serialize: SerializedExternalResource = {
    SerializedExternalResource(path, readAs.toString, options)
  }

}
object ExternalResource {
  import scala.collection.JavaConverters._
  def fromJava(path: String, readAs: String, options: java.util.HashMap[String, String]): ExternalResource = {
    ExternalResource(path, readAs, options.asScala.toMap)
  }
}
