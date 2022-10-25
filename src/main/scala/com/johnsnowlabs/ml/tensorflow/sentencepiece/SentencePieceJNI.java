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

import java.io.IOException;

class SentencePieceJNI {

    static {
        try {
            String libName = "sentencepiece_jni";
            String platform = System.getProperty("os.name").toLowerCase();
            boolean isMacOs = platform.contains("os x") || platform.contains("darwin");
            if(System.getProperty("os.arch").equals("aarch64"))
            {
                if (isMacOs) {
                    libName += "_m1";
                } else if (platform.contains("linux")) {
                    libName += "_aarch64";
                }
            }
            
            System.load(NativeLibLoader.createTempFileFromResource("/sentencepiece/" + System.mapLibraryName(libName)));
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(e.getMessage());
        }
    }

    static native long sppCtor();

    static native void sppDtor(long spp);

    static native void sppLoad(long spp, String filename) throws SentencePieceException;

    static native void sppLoadOrDie(long spp, String filename);

    static native void sppLoadFromSerializedProto(long spp, byte[] serialized) throws SentencePieceException;

    static native void sppSetEncodeExtraOptions(long spp, String extra_option) throws SentencePieceException;

    static native void sppSetDecodeExtraOptions(long spp, String extra_option) throws SentencePieceException;

    static native void sppSetVocabulary(long spp, String[] valid_vocab) throws SentencePieceException;

    static native void sppResetVocabulary(long spp) throws SentencePieceException;

    static native void sppLoadVocabulary(long spp, String filename, int threshold) throws SentencePieceException;

    static native String[] sppEncodeAsPieces(long spp, String input) throws SentencePieceException;

    static native int[] sppEncodeAsIds(long spp, String input) throws SentencePieceException;

    static native String sppDecodePieces(long spp, String[] pieces) throws SentencePieceException;

    static native String sppDecodeIds(long spp, int[] ids) throws SentencePieceException;

    static native String[][] sppNBestEncodeAsPieces(long spp, String input, int nbest_size) throws SentencePieceException;

    static native int[][] sppNBestEncodeAsIds(long spp, String input, int nbest_size) throws SentencePieceException;

    static native String[] sppSampleEncodeAsPieces(long spp, String input, int nbest_size, float alpha) throws SentencePieceException;

    static native int[] sppSampleEncodeAsIds(long spp, String input, int nbest_size, float alpha) throws SentencePieceException;

    static native byte[] sppEncodeAsSerializedProto(long spp, String input);

    static native byte[] sppSampleEncodeAsSerializedProto(long spp, String input, int nbest_size, float alpha);

    static native byte[] sppNBestEncodeAsSerializedProto(long spp, String input, int nbest_size);

    static native byte[] sppDecodePiecesAsSerializedProto(long spp, String[] pieces);

    static native byte[] sppDecodeIdsAsSerializedProto(long spp, int[] ids);

    static native int sppGetPieceSize(long spp);

    static native int sppPieceToId(long spp, String piece);

    static native String sppIdToPiece(long spp, int id);

    static native float sppGetScore(long spp, int id);

    static native boolean sppIsUnknown(long spp, int id);

    static native boolean sppIsControl(long spp, int id);

    static native boolean sppIsUnused(long spp, int id);

    static native int sppUnkId(long spp);

    static native int sppBosId(long spp);

    static native int sppEosId(long spp);

    static native int sppPadId(long spp);
}
