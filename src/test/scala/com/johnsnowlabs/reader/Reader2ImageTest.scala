/*
 * Copyright 2017-2025 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.{AnnotatorType, AssertAnnotations}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class Reader2ImageTest extends AnyFlatSpec with SparkSessionTest {

  val htmlFilesDirectory = "./src/test/resources/reader/html/"

  "Reader2Image" should "read different image source content from an HTML file" taggedAs SlowTest in {
    val sourceFile = "example-images.html"
    val reader2Image = new Reader2Image()
      .setContentType("text/html")
      .setContentPath(s"$htmlFilesDirectory/$sourceFile")
      .setOutputCol("image")

    val pipeline = new Pipeline().setStages(Array(reader2Image))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)

    val annotationsResult = AssertAnnotations.getActualImageResult(resultDf, "image")

    assert(annotationsResult.length == 2)
    annotationsResult.foreach { annotations =>
      assert(annotations.head.annotatorType == AnnotatorType.IMAGE)
      assert(annotations.head.origin == sourceFile)
      assert(annotations.head.result.nonEmpty)
      assert(annotations.head.height > 0)
      assert(annotations.head.width > 0)
      assert(annotations.head.nChannels > 0)
      assert(annotations.head.mode > 0)
      assert(annotations.head.metadata.nonEmpty)
    }

  }

}
