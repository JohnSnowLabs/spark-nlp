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

package com.johnsnowlabs.nlp

import org.apache.spark.sql.SparkSession

object SparkAccessor {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .config("spark.driver.memory", "16G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()

  val benchmarkSpark: SparkSession = SparkSession
    .builder()
    .appName("benchmark")
    .master("local[*]")
    .config("spark.driver.memory", "16G")
    .config("spark.driver.maxResultSize", "0")
    .config("spark.kryoserializer.buffer.max", "200M")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
}
