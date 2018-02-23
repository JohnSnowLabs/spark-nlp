package com.johnsnowlabs.nlp.serialization

import com.johnsnowlabs.nlp.annotators.param.SerializedAnnotatorComponent
import com.johnsnowlabs.nlp.util.io.ExternalResource

case class SerializedExternalResource(
                                       path: String,
                                       readAs: String,
                                       options: Map[String, String] = Map("format" -> "text")
                                     ) extends SerializedAnnotatorComponent[ExternalResource] {
  override def deserialize: ExternalResource = {
    ExternalResource(path, readAs, options)
  }
}
