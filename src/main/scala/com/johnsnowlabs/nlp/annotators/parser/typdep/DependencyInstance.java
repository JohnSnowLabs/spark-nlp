package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.regex.Pattern;

import static com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet.DictionaryTypes.*;

public class DependencyInstance implements Serializable {

    public enum SpecialPos {
        C, P, PNX, V, N, OTHER,
    }

    private static final long serialVersionUID = 1L;

    private int length;

    // FORM: the forms - usually words, like "thought"
    private String[] forms;

    // LEMMA: the lemmas, or stems, e.g. "think"
    private String[] lemmas;

    // COARSE-POS: the coarse part-of-speech tags, e.g."V"
    private String[] cpostags;

    // FINE-POS: the fine-grained part-of-speech tags, e.g."VBD"
    private String[] postags;

    // MOST-COARSE-POS: the coarsest part-of-speech tags (about 11 in total)
    private SpecialPos[] specialPos;

    // FEATURES: some features associated with the elements separated by "|", e.g. "PAST|3P"
    private String[][] feats;

    // HEAD: the IDs of the heads for each element
    private int[] heads;

    // DEPREL: the dependency relations, e.g. "SUBJ"
    private String[] deprels;

    public int getLength() {
        return length;
    }

    public String[] getForms() {
        return forms;
    }

    public String[] getLemmas() {
        return lemmas;
    }

    public String[] getCpostags() {
        return cpostags;
    }

    public String[] getPostags() {
        return postags;
    }

    public int[] getHeads() {
        return heads;
    }

    private int[] formids;

    public int[] getFormids() {
        return formids;
    }

    private int[] lemmaids;

    public int[] getLemmaids() {
        return lemmaids;
    }

    private int[] postagids;

    public int[] getPostagids() {
        return postagids;
    }

    private int[] cpostagids;

    public int[] getCpostagids() {
        return cpostagids;
    }

    private int[] deprelids;

    private int[][] featids;

    public int[][] getFeatids() {
        return featids;
    }

    private int[] wordVecIds;

    public int[] getWordVecIds() {
        return wordVecIds;
    }

    private int[] dependencyLabelIds;

    public int[] getDependencyLabelIds() {
        return dependencyLabelIds;
    }

    private static Pattern puncRegex = Pattern.compile("[\\p{Punct}]+", Pattern.UNICODE_CHARACTER_CLASS);

    public DependencyInstance() {}

    public DependencyInstance(int length) { this.length = length; }

    public DependencyInstance(String[] forms) {
        this.length = forms.length;
        this.forms = forms;
        this.feats = new String[length][];
        this.deprels = new String[length];
    }

    public DependencyInstance(String[] forms, String[] postags, int[] heads) {
        this.length = forms.length;
        this.forms = forms;
        this.heads = heads;
        this.postags = postags;
    }

    public DependencyInstance(String[] forms, String[] postags, int[] heads, String[] deprels) {
        this(forms, postags, heads);
        this.deprels = deprels;
    }

    public DependencyInstance(String[] forms, String[] lemmas, String[] cpostags, String[] postags,
                              String[][] feats, int[] heads, String[] deprels) {
        this(forms, postags, heads, deprels);
        this.lemmas = lemmas;
        this.feats = feats;
        this.cpostags = cpostags;
    }

    DependencyInstance(DependencyInstance dependencyInstance) {
        this.length = dependencyInstance.length;
        this.specialPos = dependencyInstance.specialPos;
        this.heads = dependencyInstance.heads;
        this.formids = dependencyInstance.formids;
        this.lemmaids = dependencyInstance.lemmaids;
        this.postagids = dependencyInstance.postagids;
        this.cpostagids = dependencyInstance.cpostagids;
        this.deprelids = dependencyInstance.deprelids;
        this.dependencyLabelIds = dependencyInstance.dependencyLabelIds;
        this.featids = dependencyInstance.featids;
        this.wordVecIds = dependencyInstance.wordVecIds;
    }

    void setInstIds(DictionarySet dicts,
                    HashMap<String, String> coarseMap, HashSet<String> conjWord) {

        formids = new int[length];
        dependencyLabelIds = new int[length];
        postagids = new int[length];
        cpostagids = new int[length];

        for (int i = 0; i < length; ++i) {
            formids[i] = dicts.lookupIndex(WORD, "form="+normalize(forms[i]));
            postagids[i] = dicts.lookupIndex(POS, "pos="+postags[i]);
            cpostagids[i] = dicts.lookupIndex(POS, "cpos="+cpostags[i]);
            dependencyLabelIds[i] = dicts.lookupIndex(DEP_LABEL, deprels[i]) - 1;	// zero-based
        }

        if (lemmas != null) {
            lemmaids = new int[length];
            for (int i = 0; i < length; ++i)
                lemmaids[i] = dicts.lookupIndex(WORD, "lemma="+normalize(lemmas[i]));
        }

        featids = new int[length][];
        for (int i = 0; i < length; ++i) if (feats[i] != null) {
            featids[i] = new int[feats[i].length];
            for (int j = 0; j < feats[i].length; ++j)
                featids[i][j] = dicts.lookupIndex(POS, "feat="+feats[i][j]);
        }

        if (dicts.getDictionarySize(WORD_VEC) > 0) {
            wordVecIds = new int[length];
            for (int i = 0; i < length; ++i) {
                int wvid = dicts.lookupIndex(WORD_VEC, forms[i]);
                if (wvid <= 0) wvid = dicts.lookupIndex(WORD_VEC, forms[i].toLowerCase());
                if (wvid > 0) wordVecIds[i] = wvid; else wordVecIds[i] = -1;
            }
        }

        // set special pos
        specialPos = new SpecialPos[length];
        for (int i = 0; i < length; ++i) {
            if (coarseMap.containsKey(postags[i])) {
                String cpos = coarseMap.get(postags[i]);
                if ((cpos.equals("CONJ")) && conjWord.contains(forms[i])) {
                    specialPos[i] = SpecialPos.C;
                }
                else if (cpos.equals("ADP"))
                    specialPos[i] = SpecialPos.P;
                else if (cpos.equals("."))
                    specialPos[i] = SpecialPos.PNX;
                else if (cpos.equals("VERB"))
                    specialPos[i] = SpecialPos.V;
                else
                    specialPos[i] = SpecialPos.OTHER;
            }
            else {
                specialPos[i] = getSpecialPos(forms[i], postags[i]);
            }
        }
    }

    private String normalize(String s) {
        if(s!=null && s.matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"))
            return "<num>";
        return s;
    }

    // Heuristic rules to "guess" POS type based on the POS tag string
    // This is an extended version of the rules in EGSTRA code
    // 	(http://groups.csail.mit.edu/nlp/egstra/).
    //
    private SpecialPos getSpecialPos(String form, String tag) {

        if (tag.charAt(0) == 'v' || tag.charAt(0) == 'V')
            return SpecialPos.V;
        else if (tag.charAt(0) == 'n' || tag.charAt(0) == 'N')
            return SpecialPos.N;
        else if (tag.equalsIgnoreCase("cc") ||
                tag.equalsIgnoreCase("conj") ||
                tag.equalsIgnoreCase("kon") ||
                tag.equalsIgnoreCase("conjunction"))
            return SpecialPos.C;
        else if (tag.equalsIgnoreCase("prep") ||
                tag.equalsIgnoreCase("preposition") ||
                tag.equals("IN"))
            return SpecialPos.P;
        else if (tag.equalsIgnoreCase("punc") ||
                tag.equals("$,") ||
                tag.equals("$.") ||
                tag.equals(",") ||
                tag.equals(";") ||
                puncRegex.matcher(form).matches())
            return SpecialPos.PNX;
        else
            return SpecialPos.OTHER;
    }

}
