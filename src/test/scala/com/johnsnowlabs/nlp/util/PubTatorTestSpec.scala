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

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.training.PubTator
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util._
import org.scalatest._

import java.io.File
import scala.reflect.io.Directory


class PubTatorTestSpec extends FlatSpec{

  "PubTator.readDataset" should "create conll-friendly dataframe" taggedAs FastTest in {

    //remove file if it's already there
    val directory = new Directory(new File("./pubtator-conll-test"))
    directory.deleteRecursively()
    val df = PubTator().readDataset(ResourceHelper.spark, "./src/test/resources/corpus_pubtator_sample.txt")
    CoNLLGenerator.exportConllFiles(df, "pubtator-conll-test")
    directory.deleteRecursively()

  }

  "PubTator.readDataset" should "create conll-friendly dataframe with not padding" taggedAs FastTest in {

    //remove file if it's already there
    val directory = new Directory(new File("./pubtator_not_padding-conll-test"))
    directory.deleteRecursively()
    val df = PubTator().readDataset(ResourceHelper.spark, "src/test/resources/corpus_pubtator_not_padding.txt",false)
    CoNLLGenerator.exportConllFiles(df, "pubtator_not_padding-conll-test")
    directory.deleteRecursively()

  }

}
