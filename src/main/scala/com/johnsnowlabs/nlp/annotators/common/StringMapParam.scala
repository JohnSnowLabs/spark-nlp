package com.johnsnowlabs.nlp.annotators.common

import org.apache.spark.ml.param._

import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._

class StringMapParam(parent: Params, name: String, doc: String, isValid: Map[String, String] => Boolean)
  extends Param[Map[String, String]](parent, name, doc, isValid) {

  def this(parent: Params, name: String, doc: String) =
    this(parent, name, doc, (_: Map[String, String]) => true)

  /** Creates a param pair with a `java.util.List` of values (for Java and Python). */
  def w(value: java.util.HashMap[String, String]): ParamPair[Map[String, String]] = w(value.asScala.toMap)

  override def jsonEncode(value: Map[String, String]): String = {
    import org.json4s.JsonDSL._
    compact(render(value.toSeq))
  }

  override def jsonDecode(json: String): Map[String, String] = {
    implicit val formats = DefaultFormats
    parse(json).extract[Seq[(String, String)]].toMap
  }

}
