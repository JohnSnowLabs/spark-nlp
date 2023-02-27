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

package com.johnsnowlabs.nlp.annotators.spell.symmetric

import org.scalatest.flatspec.AnyFlatSpec

class SymmetricDeleteModelTestSpec extends AnyFlatSpec with SymmetricDeleteBehaviors {

  "A symmetric delete approach" should behave like testDefaultTokenCorpusParameter

  "a simple isolated symmetric spell checker with a lowercase word" should behave like testSimpleCheck(
    Seq(("problex", "problem")))

  "a simple isolated symmetric spell checker with a Capitalized word" should behave like testSimpleCheck(
    Seq(("Problex", "Problem")))

  "a simple isolated symmetric spell checker with UPPERCASE word" should behave like testSimpleCheck(
    Seq(("PROBLEX", "PROBLEM")))

  "a simple isolated symmetric spell checker with noisy word" should behave like testSimpleCheck(
    Seq(("CARDIOVASCULAR:", "CARDIOVASCULAR:")))

  it should behave like testTrainUniqueWordsOnly()
  /*
    "an isolated symmetric spell checker " should behave like testSeveralChecks(Seq(
      ("problex", "problem"),
      ("contende", "continue"),
      ("pronounciation", "pronounciation"),
      ("localy", "local"),
      ("compair", "company")
    ))

    "a symmetric spell checker " should behave like testAccuracyChecks(Seq(
      ("mral", "meal"),
      ("delicatly", "delicately"),
      ("efusive", "effusive"),
      ("lauging", "laughing"),
      ("juuuuuuuuuuuuuuuussssssssssttttttttttt", "just"),
      ("screeeeeeeewed", "screwed"),
      ("readampwritepeaceee", "readampwritepeaceee")
    ))

    "a good sized dataframe with Spark pipeline" should behave like testBigPipeline

    "a good sized dataframe with Spark pipeline and spell checker dictionary" should behave like testBigPipelineDict

    "a loaded model " should behave like testLoadModel()

    "a symmetric spell checker with empty dataset" should behave like testEmptyDataset

    "A symmetric spell checker trained from fit" should behave like trainFromFitSpellChecker(Seq(
      "mral",
      "delicatly",
      "efusive",
      "lauging",
      "juuuuuuuuuuuuuuuussssssssssttttttttttt",
      "screeeeeeeewed",
      "readampwritepeaceee"
    ))

    it should behave like trainSpellCheckerModelFromFit

   */
  it should behave like raiseErrorWhenWrongColumnIsSent
}
