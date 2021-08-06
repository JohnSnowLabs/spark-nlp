/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

/**
 * This is a dictionary that contains common english abbreviations that should be considered sentence bounds
 */
object PragmaticDictionaries {

  val ABBREVIATIONS_LONG = Seq(
    "adj", "adm", "adv", "al", "ala", "alta", "apr", "arc", "ariz", "ark", "art", "assn", "asst", "attys", "aug",
    "ave", "bart", "bld", "bldg", "blvd", "brig", "bros", "btw", "cal", "calif", "capt", "cl", "cmdr", "co", "col",
    "colo", "comdr", "con", "conn", "corp", "cpl", "cres", "ct", "d.phil", "dak", "dec", "del", "dept", "det", "dist",
    "dr", "dr.phil", "dr.philos", "drs", "e.g", "ens", "esp", "esq", "etc", "exp", "expy", "ext", "feb", "fed", "fla",
    "ft", "fwy", "fy", "ga", "gen", "gov", "hon", "hosp", "hr", "hway", "hwy", "i.e", "ia", "id", "ida", "ill", "inc",
    "ind", "ing", "insp", "is", "jan", "jr", "jul", "jun", "kan", "kans", "ken", "ky", "la", "lt", "ltd", "maj", "man",
    "mar", "mass", "may", "md", "me", "med", "messrs", "mex", "mfg", "mich", "min", "minn", "miss", "mlle", "mm",
    "mme", "mo", "mont", "mr", "mrs", "ms", "msgr", "mssrs", "mt", "mtn", "neb", "nebr", "nev", "no", "nos", "nov",
    "nr", "oct", "ok", "okla", "ont", "op", "ord", "ore", "p", "pa", "pd", "pde", "penn", "penna", "pfc", "ph",
    "ph.d", "pl", "plz", "pp", "prof", "pvt", "que", "rd", "ref", "rep", "reps", "res", "rev", "rt", "sask", "sec",
    "sen", "sens", "sep", "sept", "sfc", "sgt", "sr", "st", "supt", "surg", "tce", "tenn", "tex", "univ", "usafa",
    "u.s", "ut", "va", "v", "ver", "vs", "vt", "wash", "wis", "wisc", "wy", "wyo", "yuk"
  )

  val ABBREVIATIONS = Seq(
    "adj", "bros", "btw", "co", "col", "corp", "cpl", "dec", "aug", "del", "dept", "dist",
    "dr", "e.g", "etc", "exp", "feb", "fy", "hr", "i.e", "id", "inc",
    "jan", "jr", "jul", "jun", "lt", "ltd", "mar", "may", "min", "mr", "mrs", "ms", "no", "nov",
    "oct", "ok", "p", "px", "plz", "prof", "rep", "sec",
    "sep", "sept", "sr", "st", "u.s", "ver", "vs"
  )

  val PREPOSITIVE_ABBREVIATIONS_LONG = Seq(
    "adm", "attys", "brig", "capt", "cmdr", "col", "cpl", "det", "dr", "gen", "gov", "ing", "lt", "maj", "mr", "mrs",
    "ms", "mt", "messrs", "mssrs", "prof", "ph", "rep", "reps", "rev", "sen", "sens", "sgt", "st", "supt", "v", "vs"
  )

  val PREPOSITIVE_ABBREVIATIONS = Seq(
    "dr", "mr", "ms", "prof", "mt", "st"
  )

  val NUMBER_ABBREVIATIONS_LONG = Seq(
    "art", "ext", "no", "nos", "p", "pp"
  )

  val NUMBER_ABBREVIATIONS = Seq(
    "no", "p", "px"
  )

}
