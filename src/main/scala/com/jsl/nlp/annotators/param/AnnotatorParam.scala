package com.jsl.nlp.annotators.param

import com.jsl.nlp.annotators.common.WritableAnnotatorComponent
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write

/**
  * Created by saif on 24/06/17.
  */

class AnnotatorParam[
  A <: WritableAnnotatorComponent,
  B <: SerializedAnnotatorComponent[_ <: A]]
    (identifiable: Identifiable,
      name: String,
      description: String)
    (implicit m: Manifest[B]) extends Param[A](identifiable, name, description) {

  implicit val formats = Serialization.formats(NoTypeHints)

  override def jsonEncode(value: A): String = write(value.serialize)

  override def jsonDecode(json: String): A = parse(json).extract[B].deserialize

}
