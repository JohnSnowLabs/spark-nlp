package com.johnsnowlabs.nlp.sentencepiece

import java.nio.file.{Files, Paths}

import scala.collection.JavaConverters._
import org.scalatest.{BeforeAndAfter, BeforeAndAfterEach, FlatSpec, Ignore, Tag}

class SentencePieceProcessorTestSpec extends FlatSpec with BeforeAndAfter with BeforeAndAfterEach {
  val modelPath = "./src/test/resources/sentencepiece/test_model.model"
  var sp: SentencePieceProcessor = _

  before {
    if (NativeLibLoader.getPathToNativeSentencePieceLib != null){
      sp = new SentencePieceProcessor
      sp.load(modelPath)
      sp.loadFromSerializedProto(Files.readAllBytes(Paths.get(modelPath)))
    }
  }

  after {
    if (sp != null) {
      sp.close()
    }
  }

  override def beforeEach(): Unit = {
    if (NativeLibLoader.getPathToNativeSentencePieceLib == null) {
      cancel("no sentencepiece library provided, skipping...")
    }
  }

  "the model" should "be loaded correctly" taggedAs WhenSPLibProvided in {
    assert(1000 == sp.getPieceSize)
    assert(0 == sp.pieceToId("<unk>"))
    assert(1 == sp.pieceToId("<s>"))
    assert(2 == sp.pieceToId("</s>"))
    assert("<unk>" == sp.idToPiece(0))
    assert("<s>" == sp.idToPiece(1))
    assert("</s>" == sp.idToPiece(2))
    assert(0 == sp.unkId)
    assert(1 == sp.bosId)
    assert(2 == sp.eosId)
    assert(-1 == sp.padId)
    (0 until sp.getPieceSize).foreach {
      i =>
        val piece: String = sp.idToPiece(i)
        assert(i == sp.pieceToId(piece))
    }
  }

  "the SentencePieceProcessor" should "encode in a reversible way" taggedAs WhenSPLibProvided in {
    val text: String = "I saw a girl with a telescope"
    val ids = sp.encodeAsIds(text)
    val pieces1 = sp.encodeAsPieces(text).asScala
    val pieces2 = sp.nbestEncodeAsPieces(text, 10).get(0).asScala
    assert(pieces1 == pieces2)
    assert(text == sp.decodePieces(pieces1.asJava))
    assert(text == sp.decodeIds(ids: _*))
    (0 until 100).foreach {
      _ =>
        assert(text == sp.decodePieces(sp.sampleEncodeAsPieces(text, 64, 0.5f)))
        assert(text == sp.decodePieces(sp.sampleEncodeAsPieces(text, -1, 0.5f)))
        assert(text == sp.decodeIds(sp.sampleEncodeAsIds(text, 64, 0.5f): _*))
        assert(text == sp.decodeIds(sp.sampleEncodeAsIds(text, -1, 0.5f): _*))
    }
  }

  "the SentencePieceProcessor" should "be able to serializeresults" taggedAs WhenSPLibProvided in {
    val text: String = "I saw a girl with a telescope"
    val empty = Array.emptyByteArray

    assert(!(empty sameElements sp.encodeAsSerializedProto(text)))
    assert(!(empty sameElements sp.sampleEncodeAsSerializedProto(text, 10, 0.2f)))
    assert(!(empty sameElements sp.nbestEncodeAsSerializedProto(text, 10)))
    assert(!(empty sameElements sp.decodePiecesAsSerializedProto(List("foo", "bar").asJava)))
    assert(!(empty sameElements sp.decodeIdsAsSerializedProto(20, 30)))
  }
}

object WhenSPLibProvided extends Tag(
  if (NativeLibLoader.getPathToNativeSentencePieceLib != null)
    ""
  else
    classOf[Ignore].getName
)