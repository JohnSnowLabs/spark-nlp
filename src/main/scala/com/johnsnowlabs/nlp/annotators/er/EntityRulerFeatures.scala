package com.johnsnowlabs.nlp.annotators.er

case class EntityRulerFeatures(patterns: Map[String, String], regexPatterns: Map[String, Seq[String]])
extends Serializable
