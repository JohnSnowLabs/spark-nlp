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

package com.johnsnowlabs.ml.tensorflow.sentencepiece;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


public class SentencePieceProcessor implements AutoCloseable {

    private final long rawPtr;

    public SentencePieceProcessor() {
        rawPtr = SentencePieceJNI.sppCtor();
    }

    @Override
    public void close() {
        SentencePieceJNI.sppDtor(rawPtr);
    }

    public void load(String filename) throws SentencePieceException {
        SentencePieceJNI.sppLoad(rawPtr, filename);
    }

    public void loadOrDie(String filename) {
        SentencePieceJNI.sppLoadOrDie(rawPtr, filename);
    }

    public void loadFromSerializedProto(byte[] serialized) throws SentencePieceException {
        SentencePieceJNI.sppLoadFromSerializedProto(rawPtr, serialized);
    }

    public void setEncodeExtraOptions(String extraOption) throws SentencePieceException {
        SentencePieceJNI.sppSetEncodeExtraOptions(rawPtr, extraOption);
    }

    public void setDecodeExtraOptions(String extraOption) throws SentencePieceException {
        SentencePieceJNI.sppSetDecodeExtraOptions(rawPtr, extraOption);
    }

    public void setVocabulary(List<String> validVocab) throws SentencePieceException {
        String[] bytes = validVocab.toArray(new String[0]);
        SentencePieceJNI.sppSetVocabulary(rawPtr, bytes);
    }

    public void resetVocabulary() throws SentencePieceException {
        SentencePieceJNI.sppResetVocabulary(rawPtr);
    }

    public void loadVocabulary(String filename, int threshold) throws SentencePieceException {
        SentencePieceJNI.sppLoadVocabulary(rawPtr, filename, threshold);
    }

    public List<String> encodeAsPieces(String input) throws SentencePieceException {
        String[] pieces = SentencePieceJNI.sppEncodeAsPieces(rawPtr, input);
        return Arrays.asList(pieces);
    }

    public int[] encodeAsIds(String input) throws SentencePieceException {
        return SentencePieceJNI.sppEncodeAsIds(rawPtr, input);
    }

    public String decodePieces(List<String> pieces) throws SentencePieceException {
        String[] bytes = pieces.toArray(new String[0]);
        return SentencePieceJNI.sppDecodePieces(rawPtr, bytes);
    }

    public String decodeIds(int... ids) throws SentencePieceException {
        return SentencePieceJNI.sppDecodeIds(rawPtr, ids);
    }

    public List<List<String>> nbestEncodeAsPieces(String input, int nbestSize) throws SentencePieceException {
        String[][] pieces = SentencePieceJNI.sppNBestEncodeAsPieces(rawPtr, input, nbestSize);
        return Arrays.stream(pieces)
                .map(Arrays::asList)
                .collect(Collectors.toList());
    }

    public int[][] nbestEncodeAsIds(String input, int nbestSize) throws SentencePieceException {
        return SentencePieceJNI.sppNBestEncodeAsIds(rawPtr, input, nbestSize);
    }

    public List<String> sampleEncodeAsPieces(String input, int nbestSize, float alpha) throws SentencePieceException {
        String[] pieces = SentencePieceJNI.sppSampleEncodeAsPieces(rawPtr, input, nbestSize, alpha);
        return Arrays.asList(pieces);
    }

    public int[] sampleEncodeAsIds(String input, int nbestSize, float alpha) throws SentencePieceException {
        return SentencePieceJNI.sppSampleEncodeAsIds(rawPtr, input, nbestSize, alpha);
    }

    public byte[] encodeAsSerializedProto(String input) {
        return SentencePieceJNI.sppEncodeAsSerializedProto(rawPtr, input);
    }

    public byte[] sampleEncodeAsSerializedProto(String input, int nbestSize, float alpha) {
        return SentencePieceJNI.sppSampleEncodeAsSerializedProto(rawPtr, input, nbestSize, alpha);
    }

    public byte[] nbestEncodeAsSerializedProto(String input, int nbestSize) {
        return SentencePieceJNI.sppNBestEncodeAsSerializedProto(rawPtr, input, nbestSize);
    }

    public byte[] decodePiecesAsSerializedProto(List<String> pieces) {
        String[] bytes = pieces.toArray(new String[0]);
        return SentencePieceJNI.sppDecodePiecesAsSerializedProto(rawPtr, bytes);
    }

    public byte[] decodeIdsAsSerializedProto(int... ids) {
        return SentencePieceJNI.sppDecodeIdsAsSerializedProto(rawPtr, ids);
    }

    public int getPieceSize() {
        return SentencePieceJNI.sppGetPieceSize(rawPtr);
    }

    public int pieceToId(String piece) {
        return SentencePieceJNI.sppPieceToId(rawPtr, piece);
    }

    public String idToPiece(int id) {
        return SentencePieceJNI.sppIdToPiece(rawPtr, id);
    }

    public float getScore(int id) {
        return SentencePieceJNI.sppGetScore(rawPtr, id);
    }

    public boolean isUnknown(int id) {
        return SentencePieceJNI.sppIsUnknown(rawPtr, id);
    }

    public boolean isControl(int id) {
        return SentencePieceJNI.sppIsControl(rawPtr, id);
    }

    public boolean isUnused(int id) {
        return SentencePieceJNI.sppIsUnused(rawPtr, id);
    }

    public int unkId() {
        return SentencePieceJNI.sppUnkId(rawPtr);
    }

    public int bosId() {
        return SentencePieceJNI.sppBosId(rawPtr);
    }

    public int eosId() {
        return SentencePieceJNI.sppEosId(rawPtr);
    }

    public int padId() {
        return SentencePieceJNI.sppPadId(rawPtr);
    }
}
