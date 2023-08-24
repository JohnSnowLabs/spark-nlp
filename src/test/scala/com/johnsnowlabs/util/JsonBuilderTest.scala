/*
 * Copyright 2017-2023 John Snow Labs
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

import org.scalatest.flatspec.AnyFlatSpec

class JsonBuilderTest extends AnyFlatSpec {

  val jsonTemplate =
    """
      |{
      |    "requiredField": "%s"
      |    %s
      |    %s
      |    %s
      |    %s
      |}
      |""".stripMargin

  val jsonTemplate2 =
    """
      |{
      |    "requiredField": %s
      |}
      |""".stripMargin

  "JsonBuilder" should "build a JSON format as string based on a template" in {
    val valueField1 = "myValue"
    val valueField3 = 0.78f
    val optionalField1 = JsonBuilder.formatOptionalField("optionalField1", Some(valueField1))
    val optionalField2 = JsonBuilder.formatOptionalField("optionalField2", None)
    val optionalField3 = JsonBuilder.formatOptionalField("optionalField3", Some(valueField3))
    val optionalField4 = JsonBuilder.formatOptionalField("optionalField4", None)
    val requiredFieldValue = "myRequiredValue"
    val jsonExpected =
      s"""
        |{
        |    "requiredField": "$requiredFieldValue",
        |    "optionalField1": $valueField1,
        |    "optionalField3": $valueField3
        |}
        |""".stripMargin

    val jsonResult = JsonBuilder.buildJson(
      jsonTemplate,
      requiredFieldValue,
      optionalField1,
      optionalField2,
      optionalField3,
      optionalField4)

    assert(
      jsonExpected.replaceAll("\\n", "").replaceAll(" ", "") == jsonResult
        .replaceAll("\\n", "")
        .replaceAll(" ", ""))
  }

  it should "omit optional fields with null values" in {
    val valueField1 = "myValue"
    val valueField2 = 100
    val valueField3 = 0.78f
    val valueField4 = true
    val optionalField1 = JsonBuilder.formatOptionalField("optionalField1", Some(valueField1))
    val optionalField2 = JsonBuilder.formatOptionalField("optionalField2", Some(valueField2))
    val optionalField3 = JsonBuilder.formatOptionalField("optionalField3", Some(valueField3))
    val optionalField4 = JsonBuilder.formatOptionalField("optionalField4", Some(valueField4))
    val requiredFieldValue = "myRequiredValue"
    val jsonExpected =
      s"""
         |{
         |    "requiredField": "$requiredFieldValue",
         |    "optionalField1": $valueField1,
         |    "optionalField2": $valueField2,
         |    "optionalField3": $valueField3,
         |    "optionalField4": $valueField4
         |}
         |""".stripMargin

    val jsonResult = JsonBuilder.buildJson(
      jsonTemplate,
      requiredFieldValue,
      optionalField1,
      optionalField2,
      optionalField3,
      optionalField4)

    assert(
      jsonExpected.replaceAll("\\n", "").replaceAll(" ", "") == jsonResult
        .replaceAll("\\n", "")
        .replaceAll(" ", ""))
  }

}
