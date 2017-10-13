package com.johnsnowlabs.nlp.annotators.param

import java.util.{Date, TimeZone}

import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.write

/**
  * Created by saif on 24/06/17.
  */

/**
  * Structure to guideline a writable non-standard PARAM for any annotator
  * Uses json4s jackson for serialization
  * @param identifiable Required for uid generation and storage as ML Params
  * @param name name of this Param, just like [[Param]]
  * @param description description of this Param, just like [[Param]]
  * @param m Implicit manifest constructor representation
  * @tparam A Any kind of Writable annotator component (usually a model approach)
  * @tparam B Any kind of serialized figure, of writable type
  */
class AnnotatorParam[
  A <: WritableAnnotatorComponent,
  B <: SerializedAnnotatorComponent[_ <: A]]
    (identifiable: Identifiable,
      name: String,
      description: String)
    (implicit m: Manifest[B]) extends Param[A](identifiable, name, description) {

  /** Workaround json4s issue of DefaultFormats not being Serializable in provided release*/
  object SerializableFormat extends Formats with Serializable {
    class SerializableDateFormat extends DateFormat {
      def timezone: TimeZone = throw new Exception("SerializableFormat does not implement dateformat")
      override def format(d: Date): String = throw new Exception("SerializableFormat does not implement dateformat")
      override def parse(s: String): Option[Date] = throw new Exception("SerializableFormat does not implement dateformat")
    }
    override def dateFormat: DateFormat = new SerializableDateFormat
  }

  implicit val formats = SerializableFormat

  override def jsonEncode(value: A): String = write(value.serialize)

  override def jsonDecode(json: String): A = parse(json).extract[B].deserialize
}
