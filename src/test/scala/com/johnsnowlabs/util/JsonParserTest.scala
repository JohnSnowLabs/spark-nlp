package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.annotators.er.EntityPattern
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class JsonParserTest extends AnyFlatSpec {

  "JsonParser" should "read a JSON file" in {
    import io.circe.generic.auto._
    val stream =  ResourceHelper.getResourceStream("src/test/resources/entity-ruler/pattern.json")
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]
    val expectedResult = EntityPattern("PERSON", Seq("John Snow"))

    val actualResult: EntityPattern = jsonParser.readJson(jsonContent)

    assert(actualResult == expectedResult)
  }

  it should "raise an error when JSON file has a wrong format" in {
    import io.circe.generic.auto._
    val stream =  ResourceHelper.getResourceStream("src/test/resources/entity-ruler/patterns_with_error.json")
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]

    assertThrows[UnsupportedOperationException] {
      jsonParser.readJson(jsonContent)
    }
  }

  it should "read JSON file that starts with arrays" in {
    import io.circe.generic.auto._
    val stream =  ResourceHelper.getResourceStream("src/test/resources/entity-ruler/patterns.json")
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]
    val expectedResult = Array(
      EntityPattern("PERSON", Seq("Jon", "John", "John Snow")),
      EntityPattern("PERSON", Seq("Stark", "Snow")),
      EntityPattern("PERSON", Seq("Eddard", "Eddard Stark")),
      EntityPattern("LOCATION", Seq("Winterfell")))

    val actualResult: Array[EntityPattern] = jsonParser.readJsonArray(jsonContent)

    assert(actualResult sameElements expectedResult)
  }

  it should "raise an error when reading incompatible objects" in {
    import io.circe.generic.auto._
    val stream =  ResourceHelper.getResourceStream("src/test/resources/entity-ruler/patterns.json")
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]

    assertThrows[UnsupportedOperationException] {
      jsonParser.readJson(jsonContent)
    }
  }

}
