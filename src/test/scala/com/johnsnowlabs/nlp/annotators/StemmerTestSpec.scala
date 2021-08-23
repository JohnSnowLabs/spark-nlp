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

import com.johnsnowlabs.nlp.{AnnotatorType, ContentProvider, DataBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class StemmerTestSpec extends FlatSpec with StemmerBehaviors {

  val stemmer = new Stemmer
  "a Stemmer" should s"be of type ${AnnotatorType.TOKEN}" taggedAs FastTest in {
    assert(stemmer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  val englishPhraseData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.englishPhrase)

  "A full Stemmer pipeline with English content" should behave like fullStemmerPipeline(englishPhraseData)


}
