/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.annotators.er.EntityPattern
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class JsonParserTest extends AnyFlatSpec {

  "JsonParser" should "read a JSON file" in {
    val stream =
      ResourceHelper.getResourceStream("src/test/resources/entity-ruler/single_keyword.json")
    val jsonContent = Source.fromInputStream(stream).mkString
    val expectedResult = EntityPattern("PERSON", Seq("John Snow"))

    val actualResult: EntityPattern = JsonParser.parseObject[EntityPattern](jsonContent)

    assert(actualResult == expectedResult)
  }

  it should "raise an error when JSON file has a wrong format" in {
    val stream =
      ResourceHelper.getResourceStream("src/test/resources/entity-ruler/patterns_with_error.json")
    val jsonContent = Source.fromInputStream(stream).mkString

    assertThrows[Exception] {
      JsonParser.parseObject(jsonContent)
    }
  }

  it should "read JSON file that starts with arrays" in {
    val stream =
      ResourceHelper.getResourceStream("src/test/resources/entity-ruler/keywords_only.json")
    val jsonContent = Source.fromInputStream(stream).mkString
    val expectedResult = Array(
      EntityPattern("PERSON", Seq("Jon", "John", "John Snow", "Jon Snow")),
      EntityPattern("PERSON", Seq("Stark", "Snow", "Doctor John Snow")),
      EntityPattern("PERSON", Seq("Eddard", "Eddard Stark")),
      EntityPattern("LOCATION", Seq("Winterfell")))

    val actualResult: Array[EntityPattern] = JsonParser.parseArray[EntityPattern](jsonContent)

    assert(actualResult sameElements expectedResult)
  }

  it should "raise an error when reading incompatible objects" in {
    val stream =
      ResourceHelper.getResourceStream("src/test/resources/entity-ruler/keywords_only.json")
    val jsonContent = Source.fromInputStream(stream).mkString

    assertThrows[Exception] {
      JsonParser.parseObject(jsonContent)
    }
  }

}
