package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.nlp.annotators.param.WritableAnnotatorComponent
import com.johnsnowlabs.nlp.serialization.SerializedExternalResource

/** This represents an external source which contains information into
  * how an external resource shall be read by Spark-NLP's Resource Helper.
  *  - `ReadAs.TEXT` will configure the file to be read locally as text
  *  - `ReadAs.BINARY` will configure the file to be read locally as binary
  *  - `ReadAs.SPARK` will configure the file to be read by Spark. `"format"` will need to be defined in `options`.
  *
  * ==Example==
  * {{{
  * ExternalResource(
  *   "src/test/resources/regex-matcher/rules.txt",
  *   ReadAs.TEXT,
  *   Map("delimiter" -> ",")
  * )
  *
  * ExternalResource(
  *   "src/test/resources/regex-matcher/rules.txt",
  *   ReadAs.SPARK,
  *   Map("format" -> "text", "delimiter" -> ",")
  * )
  * }}}
  *
  * @param path Path to the resource
  * @param readAs How to interpret the resource. Possible values are `ReadAs.SPARK, ReadAs.TEXT, ReadAs.BINARY`
  * @param options Options for Spark. Option `format` needs to be set if `readAs` is set to `ReadAs.SPARK`
  * */
case class ExternalResource(
                             path: String,
                             readAs: ReadAs.Format,
                             options: Map[String, String]
                           ) extends WritableAnnotatorComponent {

  if (readAs == ReadAs.SPARK)
    require(options.contains("format"), "Created ExternalResource to read as SPARK but key 'format' " +
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
