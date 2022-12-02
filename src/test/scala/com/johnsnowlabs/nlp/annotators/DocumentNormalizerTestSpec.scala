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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class DocumentNormalizerTestSpec extends AnyFlatSpec with DocumentNormalizerBehaviors {
  val documentNormalizer = new DocumentNormalizer()

  "a DocumentNormalizer output" should s"be of type ${AnnotatorType.DOCUMENT}" taggedAs FastTest in {
    assert(documentNormalizer.outputAnnotatorType == AnnotatorType.DOCUMENT)
  }
}
