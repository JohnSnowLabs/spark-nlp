/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

class EmailReaderTest extends AnyFlatSpec {

  private val spark = ResourceHelper.spark
  val emailDirectory = "src/test/resources/reader/email"

  import spark.implicits._

  "EmailReader" should "read a directory of eml files" taggedAs FastTest in {
    val emailReader = new EmailReader()
    val emailDf = emailReader.read(emailDirectory)
    emailDf.select("email").show()
    emailDf.printSchema()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
    assert(!emailDf.columns.contains("content"))
  }

  it should "read email file with attachments" taggedAs FastTest in {
    val emailFile = s"$emailDirectory/test-several-attachments.eml"
    val emailReader = new EmailReader()
    val emailDf = emailReader.read(emailFile)
    emailDf.select("email").show()

    val attachmentCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.ATTACHMENT)
      .count()
    val titleCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.TITLE)
      .count()

    val textCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.NARRATIVE_TEXT)
      .count()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
    assert(attachmentCount == 3)
    assert(titleCount == 1)
    assert(textCount == 2)
    assert(!emailDf.columns.contains("content"))
  }

  it should "read email file with two text attachments" taggedAs FastTest in {
    val emailFile = s"$emailDirectory/email-text-attachments.eml"
    val emailReader = new EmailReader()
    val emailDf = emailReader.read(emailFile)
    emailDf.select("email").show(false)
    emailDf.printSchema()

    val attachmentCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.ATTACHMENT)
      .count()
    val titleCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.TITLE)
      .count()

    val textCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.NARRATIVE_TEXT)
      .count()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
    assert(attachmentCount == 2)
    assert(titleCount == 1)
    assert(textCount == 2)
    assert(!emailDf.columns.contains("content"))
  }

  it should "read attachment content when addAttachmentContent = true" taggedAs FastTest in {
    val emailFile = s"$emailDirectory/email-text-attachments.eml"
    val emailReader = new EmailReader(addAttachmentContent = true)
    val emailDf = emailReader.read(emailFile)
    emailDf.select("email").show(false)
    emailDf.printSchema()

    val attachmentCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.ATTACHMENT)
      .count()
    val titleCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.TITLE)
      .count()

    val textCount = emailDf
      .select(explode($"email.elementType").as("elementType"))
      .filter($"elementType" === ElementType.NARRATIVE_TEXT)
      .count()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
    assert(attachmentCount == 2)
    assert(titleCount == 1)
    assert(textCount == 4)
    assert(!emailDf.columns.contains("content"))
  }

  it should "store content" taggedAs FastTest in {
    val emailReader = new EmailReader(storeContent = true)
    val emailDf = emailReader.read(emailDirectory)
    emailDf.show()

    assert(!emailDf.select(col("email").getItem(0)).isEmpty)
    assert(emailDf.columns.contains("content"))
  }

}
