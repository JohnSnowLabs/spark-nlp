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

    private static final String ROOT = "<root>";
    private static final String ROOT_LEMMA = "<root-LEMMA>";
    private static final String ROOT_POS = "<root-POS>";
    private static final String NO_TYPE = "<no-type>";

    @Override
    public DependencyInstance nextInstance() throws IOException {

        ArrayList<String[]> lines = getFileContentAsArray();
        if (lines.isEmpty()) {
            return null;
        }
        int length = lines.size();

        String[] forms = new String[length + 1];
        String[] lemmas = new String[length + 1];
        String[] uPos = new String[length + 1];
        String[] xPos = new String[length + 1];
        String[][] feats = new String[length + 1][];
        String[] deprels = new String[length + 1];
        int[] heads = new int[length + 1];

        initializeRoot(forms, lemmas, uPos, xPos, deprels, heads);
        boolean hasLemma = parseLines(lines, forms, lemmas, uPos, xPos, feats, deprels, heads);
        if (!hasLemma) lemmas = null;

        return new DependencyInstance(forms, lemmas, uPos, xPos, feats, heads, deprels, null, null);
    }

    private void initializeRoot(String[] forms, String[] lemmas, String[] uPos, String[] xPos, String[] deprels, int[] heads) {
        forms[0] = ROOT;
        lemmas[0] = ROOT_LEMMA;
        uPos[0] = ROOT_POS;
        xPos[0] = ROOT_POS;
        deprels[0] = NO_TYPE;
        heads[0] = -1;
    }

    private boolean parseLines(
        ArrayList<String[]> lines,
        String[] forms,
        String[] lemmas,
        String[] uPosList,
        String[] xPosList,
        String[][] featsMatrix,
        String[] depRels,
        int[] heads
    ) {
        boolean hasLemma = false;
        for (int i = 1; i <= lines.size(); i++) {
            String[] parts = lines.get(i - 1);
            // CoNLL Universal Dependency format
            String id = parts[0];
            String form = parts[1];
            String lemma = parts[2];
            String uPos = parts[3];
            String xPos = parts[4];
            String feats = parts[5];
            String head = parts[6];
            String depRel = parts[7];
            //String misc = parts[8];

            if (skipIteration(id, head, uPos, xPos)) continue;

            forms[i] = form;
            lemmas[i] = valueAvailable(lemma) ? lemma : null;
            hasLemma |= lemmas[i] != null;
            uPosList[i] = valueAvailable(uPos) ? uPos : xPos;
            xPosList[i] = valueAvailable(xPos) ? xPos : uPos;
            featsMatrix[i] = valueAvailable(feats) ? feats.split("\\|") : null;
            heads[i] = Integer.parseInt(head);
            depRels[i] = depRel;
        }
        return hasLemma;
    }

    private boolean skipIteration(String id, String head, String uPos, String xPos) {
        return valueIsNotNumber(id) || (valueIsNotNumber(head)) || (!valueAvailable(uPos) && !valueAvailable(xPos));
    }

    private boolean valueAvailable(String value) {
        return !value.equals("_");
    }
    private boolean valueIsNotNumber(String value) { return !value.matches("\\d+"); }


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
