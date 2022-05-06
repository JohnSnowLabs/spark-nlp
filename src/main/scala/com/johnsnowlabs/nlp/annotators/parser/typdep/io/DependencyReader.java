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

package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.ConllData;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public abstract class DependencyReader {

    private static final int END_OF_SENTENCE = -2;
    BufferedReader reader;

    public static DependencyReader createDependencyReader(String conllFormat) {
        if (conllFormat.equals("2009")) {
            return new Conll09Reader();
        } else {
            return new ConllUReader();
        }
    }

    public abstract DependencyInstance nextInstance() throws IOException;

    public DependencyInstance nextSentence(ConllData[] sentence) {

        if (sentence[0].getHead() == END_OF_SENTENCE) {
            return null;
        }

        int length = sentence.length;
        String[] forms = new String[length + 1];
        String[] lemmas = new String[length + 1];
        String[] uPos = new String[length + 1];
        String[] xPos = new String[length + 1];
        String[][] feats = new String[length + 1][];
        String[] deprels = new String[length + 1];
        int[] heads = new int[length + 1];
        int[] begins = new int[length + 1];
        int[] ends = new int[length + 1];

        forms[0] = "<root>";
        lemmas[0] = "<root-LEMMA>";
        uPos[0] = "<root-POS>";
        xPos[0] = "<root-POS>";
        deprels[0] = "<no-type>";
        heads[0] = -1;
        begins[0] = -1;
        ends[0] = -1;

        boolean hasLemma = false;

        for (int i = 1; i < length + 1; ++i) {
            ConllData conllValues = sentence[i - 1];
            begins[i] = conllValues.getBegin();
            ends[i] = conllValues.getEnd();
            forms[i] = conllValues.getForm();

            if (!conllValues.getLemma().equals("_")) {
                lemmas[i] = conllValues.getLemma();
                hasLemma = true;
            }

            uPos[i] = conllValues.getUPos();
            xPos[i] = conllValues.getXPos();

            //TODO: Add feats

            heads[i] = conllValues.getHead();
            deprels[i] = conllValues.getDepRel();

        }
        if (!hasLemma) {
            lemmas = null;
        }

        //TODO: Check why in the prediciton phase, cpos and pos are very different than training
        return new DependencyInstance(forms, lemmas, uPos, xPos, feats, heads, deprels, begins, ends);
    }

    public void startReading(String file) throws IOException {
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
    }

    public void close() throws IOException {
        if (reader != null) reader.close();
    }

}
