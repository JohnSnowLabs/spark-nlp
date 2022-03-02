package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import org.scalatest.flatspec.AnyFlatSpec

class AnnotationTest extends AnyFlatSpec {

  "Annotation" should "cast to JSON" in {
    val annotation = Annotation(TOKEN, 0, 3, "free", Map("sentence" -> "0"))
    val expectedAnnotationJson = "{\"annotatorType\":\"token\",\"begin\":0,\"end\":3,\"result\":\"free\"," +
      "\"metadata\":{\"sentence\":\"0\"},\"embeddings\":[]}"

    val annotationJson = Annotation.toJson(annotation)

    assert(expectedAnnotationJson == annotationJson)
  }

  it should "parse JSON" in {
    val annotationJson = "{\"annotatorType\":\"token\",\"begin\":0,\"end\":3,\"result\":\"free\"," +
      "\"metadata\":{\"sentence\":\"0\"},\"embeddings\":[]}"
    val expectedAnnotationJson = Annotation(TOKEN, 0, 3, "free", Map("sentence" -> "0"))

    val annotation = Annotation.parseJson(annotationJson)

    assert(expectedAnnotationJson == annotation)
  }

}
