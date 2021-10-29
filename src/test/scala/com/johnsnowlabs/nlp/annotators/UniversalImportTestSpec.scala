/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class UniversalImportTestSpec extends AnyFlatSpec {

  "A SentenceDetector" should "be imported automatically when accessing annotator pseudo package" taggedAs FastTest in {
    /** Now you can access all annotators by using this import here */
    import com.johnsnowlabs.nlp.annotator._
    require(!SparkAccessor.spark.sparkContext.isStopped)

    /** For example */
    val sentenceDetector = new SentenceDetector()
    val SSD_PATH = "./tst_shortcut_sd"
    sentenceDetector.write.overwrite().save(SSD_PATH)
    SentenceDetector.read.load(SSD_PATH)
  }

}
