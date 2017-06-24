package com.jsl.nlp.annotators.param

import com.jsl.nlp.annotators.common.AnnotatorApproach
import com.jsl.nlp.annotators.pos.perceptron.{PerceptronApproach, SerializedPerceptronApproach}
import com.jsl.nlp.annotators.sbd.pragmatic.SerializedSBDApproach
import com.jsl.nlp.annotators.sda.pragmatic.SerializedScorerApproach
import org.apache.commons.codec.{DecoderException, EncoderException}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write

/**
  * Created by saif on 24/06/17.
  */

class AnnotatorApproachParam[A <: AnnotatorApproach](identifiable: Identifiable, name: String, description: String)
  extends Param[A](identifiable, name, description) {

  implicit val formats = Serialization.formats(NoTypeHints)

  override def jsonEncode(value: A): String = {
    value match {
      case ap: AnnotatorApproach => write(ap.serialize)
      case _ => throw new EncoderException("Encode error. Unknown annotator approach")
    }
  }

  override def jsonDecode(json: String): A = {
    val rawJson = parse(json)
    (rawJson \ "id").extract[String] match {
      case SerializedPerceptronApproach.id =>
        parse(json).extract[SerializedPerceptronApproach].deserialize.asInstanceOf[A]
      case SerializedSBDApproach.id =>
        parse(json).extract[SerializedSBDApproach].deserialize.asInstanceOf[A]
      case SerializedScorerApproach.id =>
        parse(json).extract[SerializedScorerApproach].deserialize.asInstanceOf[A]
      case _ => throw new DecoderException("Decode error. Unrecognized component id")
    }
  }
}
