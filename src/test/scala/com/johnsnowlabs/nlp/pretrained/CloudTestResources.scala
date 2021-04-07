package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.util.Version

import java.sql.Timestamp


object CloudTestResources {
  val name_en_123_345_new = new ResourceMetadata(
    "name",
    Some("en"),
    Some(Version(1, 2, 3)),
    Some(Version(3, 4, 5)),
    true,
    new Timestamp(50)
  )

  val name_en_12_34_old = new ResourceMetadata(
    "name",
    Some("en"),
    Some(Version(1, 2)),
    Some(Version(3, 4)),
    true,
    new Timestamp(1)
  )

  val name_en_old = new ResourceMetadata(
    "name",
    Some("en"),
    None,
    None,
    true,
    new Timestamp(1)
  )

  val name_en_new_disabled = new ResourceMetadata(
    "name",
    Some("en"),
    None,
    None,
    false,
    new Timestamp(1)
  )

  val name_de = new ResourceMetadata(
    "name",
    Some("de"),
    None,
    None,
    true,
    new Timestamp(1)
  )
  val name_en_251_23 = new ResourceMetadata(
    "name",
    Some("en"),
    Some(Version(2, 5, 1)),
    Some(Version(2, 3)),
    true,
    new Timestamp(1)
  )
  val name_en_300_30 = new ResourceMetadata(
    "name",
    Some("en"),
    Some(Version(3, 0, 0)),
    Some(Version(3, 0)),
    true,
    new Timestamp(1)
  )
  val bert_tiny_en_300_30 = new ResourceMetadata(
    "small_bert_L2_128",
    Some("en"),
    Some(Version(3, 0, 0)),
    Some(Version(3, 0)),
    true,
    new Timestamp(1)
  )
  val all = List(name_en_123_345_new, name_en_12_34_old, name_en_old, name_en_new_disabled, name_de)
}
