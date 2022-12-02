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

import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;

import java.io.IOException;
import java.util.ArrayList;

public class ConllUReader extends DependencyReader {

    /**
     * CoNLL Universal Dependency format:
     * 0 ID
     * 1 FORM
     * 2 LEMMA
     * 3 UPOS
     * 4 XPOS
     * 5 FEATS
     * 6 HEAD
     * 7 DEPREL
     * 8 MISC
     */

    @Override
    public DependencyInstance nextInstance() throws IOException {

        ArrayList<String[]> lstLines = getFileContentAsArray();

        if (lstLines.isEmpty()) {
            return null;
        }

        int length = lstLines.size();
        String[] forms = new String[length + 1];
        String[] lemmas = new String[length + 1];
        String[] uPos = new String[length + 1];
        String[] xPos = new String[length + 1];
        String[][] feats = new String[length + 1][];
        String[] deprels = new String[length + 1];
        int[] heads = new int[length + 1];

        forms[0] = "<root>";
        lemmas[0] = "<root-LEMMA>";
        uPos[0] = "<root-POS>";
        xPos[0] = "<root-POS>";
        deprels[0] = "<no-type>";
        heads[0] = -1;

        boolean hasLemma = false;

        for (int i = 1; i < length + 1; ++i) {
            String[] parts = lstLines.get(i - 1);
            forms[i] = parts[1];
            if (!parts[2].equals("_")) {
                lemmas[i] = parts[2];
                hasLemma = true;
            }

            uPos[i] = parts[3];
            xPos[i] = parts[4];

            if (!parts[5].equals("_")) {
                feats[i] = parts[5].split("\\|");
            }

            if (parts[6].equals("_")) {
                System.out.println("Error in sentence:\n");
                System.out.println(parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3]);
            }

            heads[i] = Integer.parseInt(parts[6]);
            deprels[i] = parts[7];

        }
        if (!hasLemma) lemmas = null;

        return new DependencyInstance(forms, lemmas, uPos, xPos, feats, heads, deprels, null, null);
    }

    private ArrayList<String[]> getFileContentAsArray() throws IOException {

        ArrayList<String[]> lstLines = new ArrayList<>();

        String line = reader.readLine();
        while (line != null && !line.equals("")) {
            if (!line.startsWith("#")) {
                int endIndex = line.indexOf('\t');
                String id = line.substring(0, endIndex);
                if (!id.contains(".")) {
                    lstLines.add(line.trim().split("\t"));
                }
            }
            line = reader.readLine();
        }
        return lstLines;
    }


}
