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

package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.ContentProvider
import com.johnsnowlabs.nlp.base.{DocumentAssembler, LightPipeline, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.Benchmark
import org.scalatest._

class PragmaticDetectionPerfTest extends FlatSpec {

  "sentence detection" should "be fast" taggedAs FastTest in {

    ResourceHelper.spark
    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(true)

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        sentenceDetector
      ))

    val nermodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val nerlpmodel = new LightPipeline(nermodel)

    val data = ContentProvider.parquetData
    val n = 50000

    val subdata = data.select("text").as[String].take(n)

    Benchmark.measure(s"annotate $n sentences") {nerlpmodel.annotate(subdata)}

    val r = nerlpmodel.annotate("Hello Ms. Laura Goldman, you are always welcome here")
    println(r("sentence").mkString("##"))

  }

}
