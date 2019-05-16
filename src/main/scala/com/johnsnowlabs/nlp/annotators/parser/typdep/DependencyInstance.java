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

    // XPOS: Language-specific part-of-speech tag; underscore if not available.
    private String[] xPosTags;

    // UPOS: Universal part-of-speech tag, e.g."VBD"
    private String[] uPosTags;

    // MOST-COARSE-POS: the coarsest part-of-speech tags (about 11 in total)
    private SpecialPos[] specialPos;

    // FEATURES: some features associated with the elements separated by "|", e.g. "PAST|3P"
    private String[][] feats;

    // HEAD: the IDs of the heads for each element
    private int[] heads;

    // DEPREL: the dependency relations, e.g. "SUBJ"
    private String[] depRels;

    private int[] begins;
    private int[] ends;

    public int[] getBegins() {
        return begins;
    }

    public int[] getEnds() {
        return ends;
    }

    public int getLength() {
        return length;
    }

    public String[] getForms() {
        return forms;
    }

    public String[] getLemmas() {
        return lemmas;
    }

    public String[] getXPosTags() {
        return xPosTags;
    }

    public String[] getUPosTags() {
        return uPosTags;
    }

    public int[] getHeads() {
        return heads;
    }

    private int[] formIds;

    public int[] getFormIds() {
        return formIds;
    }

    private int[] lemmaIds;

    public int[] getLemmaIds() {
        return lemmaIds;
    }

    private int[] uPosTagIds;

    public int[] getUPosTagIds() {
        return uPosTagIds;
    }

    private int[] xPosTagIds;

    public int[] getXPosTagIds() {
        return xPosTagIds;
    }

    private int[] depRelIds;

    private int[][] featIds;

    public int[][] getFeatIds() {
        return featIds;
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
        this.depRels = new String[length];
    }

    public DependencyInstance(String[] forms, String[] uPosTags,String[] xPosTags, int[] heads) {
        this.length = forms.length;
        this.forms = forms;
        this.heads = heads;
        this.uPosTags = uPosTags;
        this.xPosTags = xPosTags;
    }

    public DependencyInstance(String[] forms, String[] uPosTags,String[] xPosTags, int[] heads, String[] depRels) {
        this(forms, uPosTags, xPosTags, heads);
        this.depRels = depRels;
    }

    public DependencyInstance(String[] forms, String[] lemmas, String[] uPosTags, String[] xPosTags,
                              String[][] feats, int[] heads, String[] depRels,
                              int[] begins, int[] ends) {
        this(forms, uPosTags, xPosTags, heads, depRels);
        this.lemmas = lemmas;
        this.feats = feats;
        this.xPosTags = xPosTags;
        this.uPosTags = uPosTags;
        this.begins = begins;
        this.ends = ends;
    }

    DependencyInstance(DependencyInstance dependencyInstance) {
        this.length = dependencyInstance.length;
        this.specialPos = dependencyInstance.specialPos;
        this.heads = dependencyInstance.heads;
        this.formIds = dependencyInstance.formIds;
        this.lemmaIds = dependencyInstance.lemmaIds;
        this.uPosTagIds = dependencyInstance.uPosTagIds;
        this.xPosTagIds = dependencyInstance.xPosTagIds;
        this.depRelIds = dependencyInstance.depRelIds;
        this.dependencyLabelIds = dependencyInstance.dependencyLabelIds;
        this.featIds = dependencyInstance.featIds;
        this.wordVecIds = dependencyInstance.wordVecIds;
    }

    void setInstIds(DictionarySet dicts,
                    HashMap<String, String> coarseMap, HashSet<String> conjWord) {

        formIds = new int[length];
        dependencyLabelIds = new int[length];
        uPosTagIds = new int[length];
        xPosTagIds = new int[length];

        for (int i = 0; i < length; ++i) {
            //TODO: Check here how the dictioaries are set
            formIds[i] = dicts.lookupIndex(WORD, "form=" + normalize(forms[i]));
            uPosTagIds[i] = dicts.lookupIndex(POS, "pos=" + uPosTags[i]);
            xPosTagIds[i] = dicts.lookupIndex(POS, "cpos=" + xPosTags[i]);
            dependencyLabelIds[i] = dicts.lookupIndex(DEP_LABEL, depRels[i]) - 1;	// zero-based
        }

        if (lemmas != null) {
            lemmaIds = new int[length];
            for (int i = 0; i < length; ++i)
                lemmaIds[i] = dicts.lookupIndex(WORD, "lemma="+normalize(lemmas[i]));
        }

        featIds = new int[length][];
        for (int i = 0; i < length; ++i) if (feats[i] != null) {
            featIds[i] = new int[feats[i].length];
            for (int j = 0; j < feats[i].length; ++j)
                featIds[i][j] = dicts.lookupIndex(POS, "feat="+feats[i][j]);
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
        //TODO: Check if this is used somewhere
        specialPos = new SpecialPos[length];
        for (int i = 0; i < length; ++i) {
            if (coarseMap.containsKey(uPosTags[i])) {
                String cpos = coarseMap.get(uPosTags[i]);
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
                specialPos[i] = getSpecialPos(forms[i], uPosTags[i]);
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
