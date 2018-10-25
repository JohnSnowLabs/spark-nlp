package com.johnsnowlabs.nlp.annotators.parser.typdep.feature;

import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Collector;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;
import com.johnsnowlabs.nlp.annotators.parser.typdep.LowRankTensor;
import com.johnsnowlabs.nlp.annotators.parser.typdep.Parameters;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Alphabet;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.FeatureVector;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.hash.TLongHashSet;

import java.io.Serializable;

import static com.johnsnowlabs.nlp.annotators.parser.typdep.feature.FeatureTemplate.Arc.*;
import static com.johnsnowlabs.nlp.annotators.parser.typdep.feature.FeatureTemplate.Word.*;

public class SyntacticFeatureFactory implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final int BITS = 30;

    private int tokenStart = 1;
    private int tokenEnd = 2;
    private int tokenMid = 3;

    public void setTokenStart(int tokenStart) {
        this.tokenStart = tokenStart;
    }

    public void setTokenEnd(int tokenEnd) {
        this.tokenEnd = tokenEnd;
    }

    public void setTokenMid(int tokenMid) {
        this.tokenMid = tokenMid;
    }

    private int tagNumBits;
    private int wordNumBits;
    private int depNumBits;
    private int flagBits;

    public int getTagNumBits() {
        return tagNumBits;
    }

    public void setTagNumBits(int tagNumBits) {
        this.tagNumBits = tagNumBits;
    }

    public int getWordNumBits() {
        return wordNumBits;
    }

    public void setWordNumBits(int wordNumBits) {
        this.wordNumBits = wordNumBits;
    }

    public int getDepNumBits() {
        return depNumBits;
    }

    public void setDepNumBits(int depNumBits) {
        this.depNumBits = depNumBits;
    }

    public int getFlagBits() {
        return flagBits;
    }

    public void setFlagBits(int flagBits) {
        this.flagBits = flagBits;
    }

    private int numberLabeledArcFeatures;
    private int numberWordFeatures;

    public int getNumberLabeledArcFeatures() {
        return numberLabeledArcFeatures;
    }

    public int getNumberWordFeatures() {
        return numberWordFeatures;
    }
    private boolean stoppedGrowth;
    private transient TLongHashSet featureHashSet;
    private Alphabet wordAlphabet;        // the alphabet of word features (e.g. \phi_h, \phi_m)

    public SyntacticFeatureFactory() {
        wordAlphabet = new Alphabet();

        stoppedGrowth = false;
        featureHashSet = new TLongHashSet(100000);

        numberWordFeatures = 0;
        numberLabeledArcFeatures = (int) ((1L << (BITS - 2)) - 1);
    }

    public void closeAlphabets() {
        wordAlphabet.stopGrowth();
        stoppedGrowth = true;
    }

    public void checkCollisions() {
        long[] codes = featureHashSet.toArray();
        int nfeats = codes.length;
        int ncols = 0;
        TIntHashSet idhash = new TIntHashSet();
        for (long code : codes) {
            int id = hashcode2int(code) & numberLabeledArcFeatures;
            if (idhash.contains(id))
                ++ncols;
            else
                idhash.add(id);
        }
        System.out.printf("Hash collision: %.4f%% (%d / %d)%n",
                ncols / (nfeats + 1e-30) * 100,
                ncols,
                nfeats
        );
    }

    /************************************************************************
     * Region start #
     *
     *  Functions that add feature codes into feature vectors and alphabets
     *
     ************************************************************************/

    static final int C1 = 0xcc9e2d51;
    static final int C2 = 0x1b873593;

    private final int hashcode2int(long code) {
        int k1 = (int) (code & 0xffffffff);
        int k2 = (int) (code >>> 32);
        int h = 0;

        k1 *= C1;
        k1 = rotl32(k1, 15, 17);
        k1 *= C2;
        h ^= k1;
        h = rotl32(h, 13, 19);
        h = h * 5 + 0xe6546b64;

        k2 *= C1;
        k2 = rotl32(k2, 15, 17);
        k2 *= C2;
        h ^= k2;
        h = rotl32(h, 13, 19);
        h = h * 5 + 0xe6546b64;

        // finalizer
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
    }

    private final int rotl32(int a, int b, int c){
        return (a << b) | (a >>> c);
    }

    private void addLabeledArcFeature(long code, Collector mat) {
        int id = hashcode2int(code) & numberLabeledArcFeatures;
        mat.addEntry(id);
        if (!stoppedGrowth)
            featureHashSet.add(code);
    }

    public void initFeatureAlphabets(DependencyInstance dependencyInstance) {
        LazyCollector lazyCollector = new LazyCollector();

        int n = dependencyInstance.getLength();

        for (int i = 0; i < n; ++i)
            createWordFeatures(dependencyInstance, i);

        for (int m = 1; m < n; ++m) {
            createLabelFeatures(lazyCollector, dependencyInstance, dependencyInstance.getHeads(),
                    dependencyInstance.getDependencyLabelIds(), m, 0);
        }
    }

    /************************************************************************
     * Region start #
     *
     *  Functions that create feature vectors of a specific word in the
     *  sentence
     *
     ************************************************************************/

    public FeatureVector createWordFeatures(DependencyInstance dependencyInstance, int i) {

        int[] pos = dependencyInstance.getPostagids();
        int[] posA = dependencyInstance.getCpostagids();
        int[] toks = dependencyInstance.getFormids();
        int[] lemma = dependencyInstance.getLemmaids();

        int p0 = pos[i];
        int pp = i > 0 ? pos[i - 1] : this.tokenStart;
        int pn = i < pos.length - 1 ? pos[i + 1] : this.tokenEnd;

        int c0 = posA[i];
        int cp = i > 0 ? posA[i - 1] : this.tokenStart;
        int cn = i < posA.length - 1 ? posA[i + 1] : this.tokenEnd;

        int w0 = toks[i];
        int wp = i == 0 ? this.tokenStart : toks[i - 1];
        int wn = i == dependencyInstance.getLength() - 1 ? this.tokenEnd : toks[i + 1];

        int l0 = 0;
        int lp = 0;
        int ln = 0;
        if (lemma != null) {
            l0 = lemma[i];
            lp = i == 0 ? this.tokenStart : lemma[i - 1];
            ln = i == dependencyInstance.getLength() - 1 ? this.tokenEnd : lemma[i + 1];
        }

        FeatureVector fv = new FeatureVector();

        long code;

        code = createWordCodeP(WORDFV_BIAS, 0);
        addWordFeature(code, fv);

        code = createWordCodeW(WORDFV_W0, w0);
        addWordFeature(code, fv);
        code = createWordCodeW(WORDFV_Wp, wp);
        addWordFeature(code, fv);
        code = createWordCodeW(WORDFV_Wn, wn);
        addWordFeature(code, fv);

        if (l0 != 0) {
            code = createWordCodeW(WORDFV_W0, l0);
            addWordFeature(code, fv);
            code = createWordCodeW(WORDFV_Wp, lp);
            addWordFeature(code, fv);
            code = createWordCodeW(WORDFV_Wn, ln);
            addWordFeature(code, fv);
        }

        code = createWordCodeP(WORDFV_P0, p0);
        addWordFeature(code, fv);
        code = createWordCodeP(WORDFV_Pp, pp);
        addWordFeature(code, fv);
        code = createWordCodeP(WORDFV_Pn, pn);
        addWordFeature(code, fv);

        code = createWordCodeP(WORDFV_P0, c0);
        addWordFeature(code, fv);
        code = createWordCodeP(WORDFV_Pp, cp);
        addWordFeature(code, fv);
        code = createWordCodeP(WORDFV_Pn, cn);
        addWordFeature(code, fv);

        code = createWordCodePP(WORDFV_PpP0, pp, p0);
        addWordFeature(code, fv);
        code = createWordCodePP(WORDFV_P0Pn, p0, pn);
        addWordFeature(code, fv);
        code = createWordCodePP(WORDFV_PpPn, pp, pn);
        addWordFeature(code, fv);
        code = createWordCodePPP(WORDFV_PpP0Pn, pp, p0, pn);
        addWordFeature(code, fv);

        code = createWordCodePP(WORDFV_PpP0, cp, c0);
        addWordFeature(code, fv);
        code = createWordCodePP(WORDFV_P0Pn, c0, cn);
        addWordFeature(code, fv);
        code = createWordCodePP(WORDFV_PpPn, cp, cn);
        addWordFeature(code, fv);
        code = createWordCodePPP(WORDFV_PpP0Pn, cp, c0, cn);
        addWordFeature(code, fv);

        code = createWordCodeWP(WORDFV_W0P0, w0, p0);
        addWordFeature(code, fv);

        code = createWordCodeWP(WORDFV_W0P0, w0, c0);
        addWordFeature(code, fv);

        if (l0 != 0) {
            code = createWordCodeWP(WORDFV_W0P0, l0, p0);
            addWordFeature(code, fv);

            code = createWordCodeWP(WORDFV_W0P0, l0, c0);
            addWordFeature(code, fv);

            code = createWordCodeWP(WORDFV_W0Pp, l0, cp);
            addWordFeature(code, fv);

            code = createWordCodeWP(WORDFV_W0Pn, l0, cn);
            addWordFeature(code, fv);

            code = createWordCodeWP(WORDFV_WpPp, lp, cp);
            addWordFeature(code, fv);

            code = createWordCodeWP(WORDFV_WnPn, ln, cn);
            addWordFeature(code, fv);
        }

        int[][] feats = dependencyInstance.getFeatids();
        if (feats[i] != null) {
            for (int u = 0; u < feats[i].length; ++u) {
                int f = feats[i][u];

                code = createWordCodeP(WORDFV_P0, f);
                addWordFeature(code, fv);

                if (l0 != 0) {
                    code = createWordCodeWP(WORDFV_W0P0, l0, f);
                    addWordFeature(code, fv);
                }

            }
        }

        return fv;
    }

    private void addWordFeature(long code, FeatureVector mat) {
        int id = wordAlphabet.lookupIndex(code, numberWordFeatures);
        if (id >= 0) {
            mat.addEntry(id);
            if (id == numberWordFeatures) ++numberWordFeatures;
        }
    }

    private void addWordFeature(long code, float value, FeatureVector mat) {
        int id = wordAlphabet.lookupIndex(code, numberWordFeatures);
        if (id >= 0) {
            mat.addEntry(id, value);
            if (id == numberWordFeatures) ++numberWordFeatures;
        }
    }

    /************************************************************************
     * Region start #
     *
     *  Functions that create feature vectors for labeled arcs
     *
     ************************************************************************/

    public void createLabelFeatures(Collector fv, DependencyInstance inst,
                                    int[] heads, int[] types, int mod, int order) {
        int head = heads[mod];
        int type = types[mod];
        if (order != 2)
            createLabeledArcFeatures(fv, inst, head, mod, type);

        int gp = heads[head];
        int ptype = types[head];
        if (order != 1 && gp != -1) {
            createLabeledGPCFeatureVector(fv, inst, gp, head, mod, type, ptype);
        }
    }

    private void createLabeledArcFeatures(Collector fv, DependencyInstance dependencyInstance, int h, int c, int type) {
        int attDist;
        attDist = h > c ? 1 : 2;

        addBasic1OFeatures(fv, dependencyInstance, h, c, attDist, type);

        addCore1OPosFeatures(fv, dependencyInstance, h, c, attDist, type);

        addCore1OBigramFeatures(fv, dependencyInstance.getFormids()[h], dependencyInstance.getPostagids()[h],
                dependencyInstance.getFormids()[c], dependencyInstance.getPostagids()[c], attDist, type);

        if (dependencyInstance.getLemmaids() != null)
            addCore1OBigramFeatures(fv, dependencyInstance.getLemmaids()[h], dependencyInstance.getPostagids()[h],
                    dependencyInstance.getLemmaids()[c], dependencyInstance.getPostagids()[c], attDist, type);

        addCore1OBigramFeatures(fv, dependencyInstance.getFormids()[h], dependencyInstance.getCpostagids()[h],
                dependencyInstance.getFormids()[c], dependencyInstance.getCpostagids()[c], attDist, type);

        if (dependencyInstance.getLemmaids() != null)
            addCore1OBigramFeatures(fv, dependencyInstance.getLemmaids()[h], dependencyInstance.getCpostagids()[h],
                    dependencyInstance.getLemmaids()[c], dependencyInstance.getCpostagids()[c], attDist, type);

        if (dependencyInstance.getFeatids()[h] != null && dependencyInstance.getFeatids()[c] != null) {
            for (int i = 0, N = dependencyInstance.getFeatids()[h].length; i < N; ++i)
                for (int j = 0, M = dependencyInstance.getFeatids()[c].length; j < M; ++j) {

                    addCore1OBigramFeatures(fv, dependencyInstance.getFormids()[h], dependencyInstance.getFeatids()[h][i],
                            dependencyInstance.getFormids()[c], dependencyInstance.getFeatids()[c][j], attDist, type);

                    if (dependencyInstance.getLemmas() != null)
                        addCore1OBigramFeatures(fv, dependencyInstance.getLemmaids()[h], dependencyInstance.getFeatids()[h][i],
                                dependencyInstance.getLemmaids()[c], dependencyInstance.getFeatids()[c][j], attDist, type);
                }
        }

    }

    private void addBasic1OFeatures(Collector fv, DependencyInstance dependencyInstance,
                                    int h, int m, int attDist, int type) {

        long code;            // feature code

        int[] forms = dependencyInstance.getFormids();
        int[] lemmas = dependencyInstance.getLemmaids();
        int[] postags = dependencyInstance.getPostagids();
        int[] cpostags = dependencyInstance.getCpostagids();
        int[][] feats = dependencyInstance.getFeatids();

        int tid = type << 4;

        code = createArcCodeW(CORE_HEAD_WORD, forms[h]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeW(CORE_MOD_WORD, forms[m]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWW(HW_MW, forms[h], forms[m]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        int pHF;
        if (h == 0) pHF = this.tokenStart;
        else pHF = h == m + 1 ? this.tokenMid : forms[h - 1];

        int nHF;
        if (h == dependencyInstance.getLength() - 1) nHF = this.tokenEnd;
        else nHF = h + 1 == m ? this.tokenMid : forms[h + 1];

        int pMF;
        if (m == 0) pMF = this.tokenStart;
        else pMF = m == h + 1 ? this.tokenMid : forms[m - 1];

        int nMF;
        if (m == dependencyInstance.getLength() - 1) nMF = this.tokenEnd;
        else nMF = m + 1 == h ? this.tokenMid : forms[m + 1];

        code = createArcCodeW(CORE_HEAD_pWORD, pHF) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeW(CORE_HEAD_nWORD, nHF) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeW(CORE_MOD_pWORD, pMF) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeW(CORE_MOD_nWORD, nMF) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);


        code = createArcCodeP(CORE_HEAD_POS, postags[h]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeP(CORE_HEAD_POS, cpostags[h]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeP(CORE_MOD_POS, postags[m]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeP(CORE_MOD_POS, cpostags[m]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(HP_MP, postags[h], postags[m]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(HP_MP, cpostags[h], cpostags[m]) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);


        if (lemmas != null) {
            code = createArcCodeW(CORE_HEAD_WORD, lemmas[h]) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);

            code = createArcCodeW(CORE_MOD_WORD, lemmas[m]) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);

            code = createArcCodeWW(HW_MW, lemmas[h], lemmas[m]) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);

            int pHL;
            if (h == 0) pHL = this.tokenStart;
            else pHL = h == m + 1 ? this.tokenMid : lemmas[h - 1];

            int nHL;
            if (h == dependencyInstance.getLength() - 1) nHL = this.tokenEnd;
            else nHL = h + 1 == m ? this.tokenMid : lemmas[h + 1];

            int pML;
            if (m == 0) pML = this.tokenStart;
            else pML = m == h + 1 ? this.tokenMid : lemmas[m - 1];

            int nML;
            if (m == dependencyInstance.getLength() - 1) nML = this.tokenEnd;
            else nML = m + 1 == h ? this.tokenMid : lemmas[m + 1];

            code = createArcCodeW(CORE_HEAD_pWORD, pHL) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);

            code = createArcCodeW(CORE_HEAD_nWORD, nHL) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);

            code = createArcCodeW(CORE_MOD_pWORD, pML) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);

            code = createArcCodeW(CORE_MOD_nWORD, nML) | tid;
            addLabeledArcFeature(code, fv);
            addLabeledArcFeature(code | attDist, fv);
        }

        if (feats[h] != null)
            for (int i = 0, N = feats[h].length; i < N; ++i) {
                code = createArcCodeP(CORE_HEAD_POS, feats[h][i]) | tid;
                addLabeledArcFeature(code, fv);
                addLabeledArcFeature(code | attDist, fv);
            }

        if (feats[m] != null)
            for (int i = 0, N = feats[m].length; i < N; ++i) {
                code = createArcCodeP(CORE_MOD_POS, feats[m][i]) | tid;
                addLabeledArcFeature(code, fv);
                addLabeledArcFeature(code | attDist, fv);
            }

        if (feats[h] != null && feats[m] != null) {
            for (int i = 0, N = feats[h].length; i < N; ++i)
                for (int j = 0, M = feats[m].length; j < M; ++j) {
                    code = createArcCodePP(HP_MP, feats[h][i], feats[m][j]) | tid;
                    addLabeledArcFeature(code, fv);
                    addLabeledArcFeature(code | attDist, fv);
                }
        }

    }

    private void addCore1OPosFeatures(Collector fv, DependencyInstance dependencyInstance,
                                      int h, int c, int attDist, int type) {

        int[] pos = dependencyInstance.getPostagids();
        int[] posA = dependencyInstance.getCpostagids();

        int tid = type << 4;

        int pHead = pos[h];
        int pHeadA = posA[h];
        int pMod = pos[c];
        int pModA = posA[c];

        int pHeadLeft;
        if (h > 0) pHeadLeft = h - 1 == c ? this.tokenMid : pos[h - 1];
        else pHeadLeft = this.tokenStart;

        int pModRight;
        if (c < pos.length - 1) pModRight = c + 1 == h ? this.tokenMid : pos[c + 1];
        else pModRight = this.tokenEnd;

        int pHeadRight;
        if (h < pos.length - 1) pHeadRight = h + 1 == c ? this.tokenMid : pos[h + 1];
        else pHeadRight = this.tokenEnd;

        int pModLeft;
        if (c > 0) pModLeft = c - 1 == h ? this.tokenMid : pos[c - 1];
        else pModLeft = this.tokenStart;

        int pHeadLeftA;
        if (h > 0) pHeadLeftA = h - 1 == c ? this.tokenMid : posA[h - 1];
        else pHeadLeftA = this.tokenStart;

        int pModRightA;
        if (c < posA.length - 1) pModRightA = c + 1 == h ? this.tokenMid : posA[c + 1];
        else pModRightA = this.tokenEnd;

        int pHeadRightA;
        if (h < posA.length - 1) pHeadRightA = h + 1 == c ? this.tokenMid : posA[h + 1];
        else pHeadRightA = this.tokenEnd;

        int pModLeftA;
        if (c > 0) pModLeftA = c - 1 == h ? this.tokenMid : posA[c - 1];
        else pModLeftA = this.tokenStart;


        long code;

        // feature posR posMid posL
        code = createArcCodePP(HPp_HP, pHeadLeft, pHead) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(HP_HPn, pHead, pHeadRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_HP_HPn, pHeadLeft, pHead, pHeadRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(MPp_MP, pModLeft, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(MP_MPn, pMod, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(MPp_MP_MPn, pModLeft, pMod, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(HPp_HP, pHeadLeftA, pHeadA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(HP_HPn, pHeadA, pHeadRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_HP_HPn, pHeadLeftA, pHeadA, pHeadRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(MPp_MP, pModLeftA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePP(MP_MPn, pModA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(MPp_MP_MPn, pModLeftA, pModA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPPP(HPp_HP_MP_MPn, pHeadLeft, pHead, pMod, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_MP_MPn, pHead, pMod, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_HP_MP, pHeadLeft, pHead, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_MP_MPn, pHeadLeft, pMod, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_HP_MPn, pHeadLeft, pHead, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPPP(HPp_HP_MP_MPn, pHeadLeftA, pHeadA, pModA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_MP_MPn, pHeadA, pModA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_HP_MP, pHeadLeftA, pHeadA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_MP_MPn, pHeadLeftA, pModA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPp_HP_MPn, pHeadLeftA, pHeadA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);


        // feature posL posL+1 posR-1 posR
        code = createArcCodePPPP(HP_HPn_MPp_MP, pHead, pHeadRight, pModLeft, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_MPp_MP, pHead, pModLeft, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_HPn_MP, pHead, pHeadRight, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPn_MPp_MP, pHeadRight, pModLeft, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_HPn_MPp, pHead, pHeadRight, pModLeft) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPPP(HP_HPn_MPp_MP, pHeadA, pHeadRightA, pModLeftA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_MPp_MP, pHeadA, pModLeftA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_HPn_MP, pHeadA, pHeadRightA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HPn_MPp_MP, pHeadRightA, pModLeftA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPP(HP_HPn_MPp, pHeadA, pHeadRightA, pModLeftA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);


        // feature posL-1 posL posR-1 posR
        // feature posL posL+1 posR posR+1
        code = createArcCodePPPP(HPp_HP_MPp_MP, pHeadLeft, pHead, pModLeft, pMod) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPPP(HP_HPn_MP_MPn, pHead, pHeadRight, pMod, pModRight) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPPP(HPp_HP_MPp_MP, pHeadLeftA, pHeadA, pModLeftA, pModA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodePPPP(HP_HPn_MP_MPn, pHeadA, pHeadRightA, pModA, pModRightA) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

    }

    private void addCore1OBigramFeatures(Collector fv, int head, int headP,
                                         int mod, int modP, int attDist, int type) {

        long code;

        int tid = type << 4;

        code = createArcCodeWWPP(HW_MW_HP_MP, head, mod, headP, modP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWPP(MW_HP_MP, mod, headP, modP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWPP(HW_HP_MP, head, headP, modP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWP(MW_HP, mod, headP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWP(HW_MP, head, modP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWP(HW_HP, head, headP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

        code = createArcCodeWP(MW_MP, mod, modP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | attDist, fv);

    }

    private void createLabeledGPCFeatureVector(Collector fv, DependencyInstance dependencyInstance,
                                               int gp, int par, int c, int type, int ptype) {

        int[] pos = dependencyInstance.getPostagids();
        int[] posA = dependencyInstance.getCpostagids();
        int[] lemma = dependencyInstance.getLemmaids() != null ? dependencyInstance.getLemmaids() :
                dependencyInstance.getFormids();

        int flag = (((((gp > par ? 0 : 1) << 1) | (par > c ? 0 : 1)) << 1) | 1);
        int tid = ((ptype << depNumBits) | type) << 4;

        int GP = pos[gp];
        int HP = pos[par];
        int MP = pos[c];
        int GC = posA[gp];
        int HC = posA[par];
        int MC = posA[c];
        long code;

        code = createArcCodePPP(GP_HP_MP, GP, HP, MP) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodePPP(GC_HC_MC, GC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        int GL = lemma[gp];
        int HL = lemma[par];
        int ML = lemma[c];

        code = createArcCodeWPP(GL_HC_MC, GL, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWPP(GC_HL_MC, HL, GC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWPP(GC_HC_ML, ML, GC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodePP(GC_HC, GC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodePP(GC_MC, GC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodePP(HC_MC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWP(GL_HL_MC, GL, HL, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWP(GL_HC_ML, GL, ML, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWP(GC_HL_ML, HL, ML, GC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWW(GL_HL_ML, GL, HL, ML) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GL_HC, GL, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GC_HL, HL, GC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWW(GL_HL, GL, HL) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GL_MC, GL, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GC_ML, ML, GC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWW(GL_ML, GL, ML) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(HL_MC, HL, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(HC_ML, ML, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWW(HL_ML, HL, ML) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        addLabeledTurboGPC(dependencyInstance, gp, par, c, flag, tid, fv);
    }

    private void addLabeledTurboGPC(DependencyInstance dependencyInstance, int gp, int par, int c,
                                    int dirFlag, int tid, Collector fv) {
        int[] posA = dependencyInstance.getCpostagids();
        int[] lemma = dependencyInstance.getLemmaids() != null ? dependencyInstance.getLemmaids() :
                dependencyInstance.getFormids();
        int len = posA.length;

        int GC = posA[gp];
        int HC = posA[par];
        int MC = posA[c];

        int pGC = gp > 0 ? posA[gp - 1] : this.tokenStart;
        int nGC = gp < len - 1 ? posA[gp + 1] : this.tokenEnd;
        int pHC = par > 0 ? posA[par - 1] : this.tokenStart;
        int nHC = par < len - 1 ? posA[par + 1] : this.tokenEnd;
        int pMC = c > 0 ? posA[c - 1] : this.tokenStart;
        int nMC = c < len - 1 ? posA[c + 1] : this.tokenEnd;

        long code = 0;

        // CCC
        code = createArcCodePPPP(pGC_GC_HC_MC, pGC, GC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPP(GC_nGC_HC_MC, GC, nGC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPP(GC_pHC_HC_MC, GC, pHC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPP(GC_HC_nHC_MC, GC, HC, nHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPP(GC_HC_pMC_MC, GC, HC, pMC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPP(GC_HC_MC_nMC, GC, HC, MC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_pGC_pHC, GC, HC, MC, pGC, pHC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_pGC_pMC, GC, HC, MC, pGC, pMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_pHC_pMC, GC, HC, MC, pHC, pMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_nGC_nHC, GC, HC, MC, nGC, nHC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_nGC_nMC, GC, HC, MC, nGC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_nHC_nMC, GC, HC, MC, nHC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_pGC_nHC, GC, HC, MC, pGC, nHC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_pGC_nMC, GC, HC, MC, pGC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_pHC_nMC, GC, HC, MC, pHC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_nGC_pHC, GC, HC, MC, nGC, pHC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_nGC_pMC, GC, HC, MC, nGC, pMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodePPPPP(GC_HC_MC_nHC_pMC, GC, HC, MC, nHC, pMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        int GL = lemma[gp];
        int HL = lemma[par];
        int ML = lemma[c];

        // LCC
        code = createArcCodeWPPP(pGC_GL_HC_MC, GL, pGC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_nGC_HC_MC, GL, nGC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_pHC_HC_MC, GL, pHC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_HC_nHC_MC, GL, HC, nHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_HC_pMC_MC, GL, HC, pMC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_HC_MC_nMC, GL, HC, MC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        // CLC
        code = createArcCodeWPPP(pGC_GC_HL_MC, HL, pGC, GC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_nGC_HL_MC, HL, GC, nGC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_pHC_HL_MC, HL, GC, pHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HL_nHC_MC, HL, GC, nHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HL_pMC_MC, HL, GC, pMC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HL_MC_nMC, HL, GC, MC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        // CCL
        code = createArcCodeWPPP(pGC_GC_HC_ML, ML, pGC, GC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_nGC_HC_ML, ML, GC, nGC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_pHC_HC_ML, ML, GC, pHC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HC_nHC_ML, ML, GC, HC, nHC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HC_pMC_ML, ML, GC, HC, pMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HC_ML_nMC, ML, GC, HC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);
    }


    /************************************************************************
     * Region start #
     *
     *  Functions to create or parse 64-bit feature code
     *
     *  A feature code is like:
     *
     *    X1 X2 .. Xk TEMP DIST
     *
     *  where Xi   is the integer id of a word, pos tag, etc.
     *        TEMP is the integer id of the feature template
     *        DIST is the integer binned length  (4 bits)
     ************************************************************************/

    private long extractArcTemplateCode(long code) {
        return (code >> flagBits) & ((1 << NUM_ARC_FEAT_BITS) - 1);
    }

    private long extractDistanceCode(long code) {
        return code & 15;
    }

    private long extractLabelCode(long code) {
        return (code >> 4) & ((1 << depNumBits) - 1);
    }

    private long extractPLabelCode(long code) {
        return (code >> (depNumBits + 4)) & ((1 << depNumBits) - 1);
    }

    private void extractArcCodeP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[0] = (int) (code & ((1 << tagNumBits) - 1));
    }

    private void extractArcCodePP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << tagNumBits) - 1));
    }

    private void extractArcCodePPP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << tagNumBits) - 1));
    }

    private void extractArcCodePPPP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[3] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << tagNumBits) - 1));
    }

    private void extractArcCodePPPPP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[4] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[3] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << tagNumBits) - 1));
    }

    private void extractArcCodeWPPP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[3] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeW(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeWW(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[1] = (int) (code & ((1 << wordNumBits) - 1));
        code = code >> wordNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeWP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeWPP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeWWP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << wordNumBits) - 1));
        code = code >> wordNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeWWW(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[2] = (int) (code & ((1 << wordNumBits) - 1));
        code = code >> wordNumBits;
        x[1] = (int) (code & ((1 << wordNumBits) - 1));
        code = code >> wordNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private void extractArcCodeWWPP(long code, int[] x) {
        code = (code >> flagBits) >> NUM_ARC_FEAT_BITS;
        x[3] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[2] = (int) (code & ((1 << tagNumBits) - 1));
        code = code >> tagNumBits;
        x[1] = (int) (code & ((1 << wordNumBits) - 1));
        code = code >> wordNumBits;
        x[0] = (int) (code & ((1 << wordNumBits) - 1));
    }

    private long createArcCodeP(FeatureTemplate.Arc temp, long x) {
        return ((x << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodePP(FeatureTemplate.Arc temp, long x, long y) {
        return ((((x << tagNumBits) | y) << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodePPP(FeatureTemplate.Arc temp, long x, long y, long z) {
        return ((((((x << tagNumBits) | y) << tagNumBits) | z) << NUM_ARC_FEAT_BITS)
                | temp.ordinal()) << flagBits;
    }

    private long createArcCodePPPP(FeatureTemplate.Arc temp, long x, long y, long u, long v) {
        return ((((((((x << tagNumBits) | y) << tagNumBits) | u) << tagNumBits) | v)
                << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodePPPPP(FeatureTemplate.Arc temp, long x, long y, long u, long v, long w) {
        return ((((((((((x << tagNumBits) | y) << tagNumBits) | u) << tagNumBits) | v) << tagNumBits) | w)
                << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodeW(FeatureTemplate.Arc temp, long x) {
        return ((x << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWW(FeatureTemplate.Arc temp, long x, long y) {
        return ((((x << wordNumBits) | y) << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWWW(FeatureTemplate.Arc temp, long x, long y, long z) {
        return ((((((x << wordNumBits) | y) << wordNumBits) | z) << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWP(FeatureTemplate.Arc temp, long x, long y) {
        return ((((x << tagNumBits) | y) << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWPP(FeatureTemplate.Arc temp, long x, long y, long z) {
        return ((((((x << tagNumBits) | y) << tagNumBits) | z) << NUM_ARC_FEAT_BITS)
                | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWPPP(FeatureTemplate.Arc temp, long x, long y, long u, long v) {
        return ((((((((x << tagNumBits) | y) << tagNumBits) | u) << tagNumBits) | v) << NUM_ARC_FEAT_BITS)
                | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWWP(FeatureTemplate.Arc temp, long x, long y, long z) {
        return ((((((x << wordNumBits) | y) << tagNumBits) | z) << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createArcCodeWWPP(FeatureTemplate.Arc temp, long x, long y, long u, long v) {
        return ((((((((x << wordNumBits) | y) << tagNumBits) | u) << tagNumBits) | v)
                << NUM_ARC_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createWordCodeW(FeatureTemplate.Word temp, long x) {
        return ((x << NUM_WORD_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createWordCodeP(FeatureTemplate.Word temp, long x) {
        return ((x << NUM_WORD_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createWordCodePP(FeatureTemplate.Word temp, long x, long y) {
        return ((((x << tagNumBits) | y) << NUM_WORD_FEAT_BITS) | temp.ordinal()) << flagBits;
    }

    private long createWordCodePPP(FeatureTemplate.Word temp, long x, long y, long z) {
        return ((((((x << tagNumBits) | y) << tagNumBits) | z) << NUM_WORD_FEAT_BITS)
                | temp.ordinal()) << flagBits;
    }

    private long createWordCodeWP(FeatureTemplate.Word temp, long x, long y) {
        return ((((x << tagNumBits) | y) << NUM_WORD_FEAT_BITS) | temp.ordinal()) << flagBits;
    }


    /**********************************************************************
     *  Region end #
     ************************************************************************/

    private void clearFeatureHashSet() {
        featureHashSet = null;
    }

    public void fillParameters(LowRankTensor tensor, LowRankTensor tensor2, Parameters params) {

        long[] codes = featureHashSet.toArray();
        clearFeatureHashSet();
        int[] x = new int[5];

        for (long code : codes) {

            int dist = (int) extractDistanceCode(code);
            int temp = (int) extractArcTemplateCode(code);

            int label = (int) extractLabelCode(code);
            int plabel = (int) extractPLabelCode(code);

            long head = -1;
            long mod = -1;
            long gp = -1;

            if (temp == HPp_HP.ordinal()) {
                extractArcCodePP(code, x);
            }

            if (temp == HP_HPn.ordinal()) {
                extractArcCodePP(code, x);
            }

            if (temp == HPp_HP_HPn.ordinal()) {
                extractArcCodePPP(code, x);
            }

            if (temp == MPp_MP.ordinal()) {
                extractArcCodePP(code, x);
            }

            if (temp == MP_MPn.ordinal()) {
                extractArcCodePP(code, x);
            }

            if (temp == MPp_MP_MPn.ordinal()) {
                extractArcCodePPP(code, x);
            }

            if (temp == HPp_HP_MP_MPn.ordinal()) {
                extractArcCodePPPP(code, x);
                head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
            } else if (temp == HP_MP_MPn.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodeP(WORDFV_P0, x[0]);
                mod = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
            } else if (temp == HPp_HP_MP.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == HPp_MP_MPn.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodeP(WORDFV_Pp, x[0]);
                mod = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
            } else if (temp == HPp_HP_MPn.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
                mod = createWordCodeP(WORDFV_Pn, x[2]);
            } else if (temp == HP_HPn_MPp_MP.ordinal()) {
                extractArcCodePPPP(code, x);
                head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
            } else if (temp == HP_MPp_MP.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodeP(WORDFV_P0, x[0]);
                mod = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
            } else if (temp == HP_HPn_MP.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == HPn_MPp_MP.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodeP(WORDFV_Pn, x[0]);
                mod = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
            } else if (temp == HP_HPn_MPp.ordinal()) {
                extractArcCodePPP(code, x);
                head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
                mod = createWordCodeP(WORDFV_Pp, x[2]);
            } else if (temp == HPp_HP_MPp_MP.ordinal()) {
                extractArcCodePPPP(code, x);
                head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
            } else if (temp == HP_HPn_MP_MPn.ordinal()) {
                extractArcCodePPPP(code, x);
                head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
            } else if (temp == HW_MW_HP_MP.ordinal()) {
                extractArcCodeWWPP(code, x);
                head = createWordCodeWP(WORDFV_W0P0, x[0], x[2]);
                mod = createWordCodeWP(WORDFV_W0P0, x[1], x[3]);
            } else if (temp == MW_HP_MP.ordinal()) {
                extractArcCodeWPP(code, x);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeWP(WORDFV_W0P0, x[0], x[2]);
            } else if (temp == HW_HP_MP.ordinal()) {
                extractArcCodeWPP(code, x);
                head = createWordCodeWP(WORDFV_W0P0, x[0], x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == MW_HP.ordinal()) {
                extractArcCodeWP(code, x);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == HW_MP.ordinal()) {
                extractArcCodeWP(code, x);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_P0, x[1]);
            } else if (temp == HW_MW.ordinal()) {
                extractArcCodeWW(code, x);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeW(WORDFV_W0, x[1]);
            } else if (temp == HP_MP.ordinal()) {
                extractArcCodePP(code, x);
                head = createWordCodeW(WORDFV_P0, x[0]);
                mod = createWordCodeW(WORDFV_P0, x[1]);
            } else if (temp == HW_HP.ordinal()) {
                extractArcCodeWP(code, x);
                head = createWordCodeWP(WORDFV_W0P0, x[0], x[1]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == MW_MP.ordinal()) {
                extractArcCodeWP(code, x);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeWP(WORDFV_W0P0, x[0], x[1]);
            } else if (temp == CORE_HEAD_WORD.ordinal()) {
                extractArcCodeW(code, x);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == CORE_HEAD_POS.ordinal()) {
                extractArcCodeP(code, x);
                head = createWordCodeP(WORDFV_P0, x[0]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == CORE_MOD_WORD.ordinal()) {
                extractArcCodeW(code, x);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == CORE_MOD_POS.ordinal()) {
                extractArcCodeP(code, x);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeP(WORDFV_P0, x[0]);
            } else if (temp == CORE_HEAD_pWORD.ordinal()) {
                extractArcCodeW(code, x);
                head = createWordCodeW(WORDFV_Wp, x[0]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == CORE_HEAD_nWORD.ordinal()) {
                extractArcCodeW(code, x);
                head = createWordCodeW(WORDFV_Wn, x[0]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == CORE_MOD_pWORD.ordinal()) {
                extractArcCodeW(code, x);
                mod = createWordCodeW(WORDFV_Wp, x[0]);
                head = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == CORE_MOD_nWORD.ordinal()) {
                extractArcCodeW(code, x);
                mod = createWordCodeW(WORDFV_Wn, x[0]);
                head = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == HEAD_EMB.ordinal()) {
                extractArcCodeW(code, x);
                head = createWordCodeW(WORDFV_EMB, x[0]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == MOD_EMB.ordinal()) {
                extractArcCodeW(code, x);
                mod = createWordCodeW(WORDFV_EMB, x[0]);
                head = createWordCodeP(WORDFV_BIAS, 0);
            }

            // second order
            else if (temp == GP_HP_MP.ordinal() || temp == GC_HC_MC.ordinal()) {
                extractArcCodePPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GL_HC_MC.ordinal()) {
                extractArcCodeWPP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GC_HL_MC.ordinal()) {
                extractArcCodeWPP(code, x);        // HL, GC, MC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GC_HC_ML.ordinal()) {
                extractArcCodeWPP(code, x);        // ML, GC, HC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == GL_HL_MC.ordinal()) {
                extractArcCodeWWP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeW(WORDFV_W0, x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GL_HC_ML.ordinal()) {
                extractArcCodeWWP(code, x);        // GL, ML, HC
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeW(WORDFV_W0, x[1]);
            } else if (temp == GC_HL_ML.ordinal()) {
                extractArcCodeWWP(code, x);        // HL, ML, GC
                gp = createWordCodeP(WORDFV_P0, x[2]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeW(WORDFV_W0, x[1]);
            } else if (temp == GL_HL_ML.ordinal()) {
                extractArcCodeWWW(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeW(WORDFV_W0, x[1]);
                mod = createWordCodeW(WORDFV_W0, x[2]);
            } else if (temp == GC_HC.ordinal()) {
                extractArcCodePP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == GL_HC.ordinal()) {
                extractArcCodeWP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == GC_HL.ordinal()) {
                extractArcCodeWP(code, x);        // HL, GC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == GL_HL.ordinal()) {
                extractArcCodeWW(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeW(WORDFV_W0, x[1]);
                mod = createWordCodeP(WORDFV_BIAS, 0);
            } else if (temp == GC_MC.ordinal()) {
                extractArcCodePP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeP(WORDFV_P0, x[1]);
            } else if (temp == GL_MC.ordinal()) {
                extractArcCodeWP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeP(WORDFV_P0, x[1]);
            } else if (temp == GC_ML.ordinal()) {
                extractArcCodeWP(code, x);        // ML, GC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == GL_ML.ordinal()) {
                extractArcCodeWW(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_BIAS, 0);
                mod = createWordCodeW(WORDFV_W0, x[1]);
            } else if (temp == HC_MC.ordinal()) {
                extractArcCodePP(code, x);
                gp = createWordCodeP(WORDFV_BIAS, 0);
                head = createWordCodeP(WORDFV_P0, x[0]);
                mod = createWordCodeP(WORDFV_P0, x[1]);
            } else if (temp == HL_MC.ordinal()) {
                extractArcCodeWP(code, x);
                gp = createWordCodeP(WORDFV_BIAS, 0);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_P0, x[1]);
            } else if (temp == HC_ML.ordinal()) {
                extractArcCodeWP(code, x);        // ML, HC
                gp = createWordCodeP(WORDFV_BIAS, 0);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == HL_ML.ordinal()) {
                extractArcCodeWW(code, x);
                gp = createWordCodeP(WORDFV_BIAS, 0);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeW(WORDFV_W0, x[1]);
            } else if (temp == pGC_GC_HC_MC.ordinal()) {
                extractArcCodePPPP(code, x);
                gp = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_nGC_HC_MC.ordinal()) {
                extractArcCodePPPP(code, x);
                gp = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_pHC_HC_MC.ordinal()) {
                extractArcCodePPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_HC_nHC_MC.ordinal()) {
                extractArcCodePPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_HC_pMC_MC.ordinal()) {
                extractArcCodePPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
            } else if (temp == GC_HC_MC_nMC.ordinal()) {
                extractArcCodePPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
            } else if (temp == pGC_GL_HC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);        // GL, pGC, HC, MC
                gp = createWordCodeWP(WORDFV_W0Pp, x[0], x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GL_nGC_HC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);
                gp = createWordCodeWP(WORDFV_W0Pn, x[0], x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GL_pHC_HC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GL_HC_nHC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GL_HC_pMC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
            } else if (temp == GL_HC_MC_nMC.ordinal()) {
                extractArcCodeWPPP(code, x);
                gp = createWordCodeW(WORDFV_W0, x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
            } else if (temp == pGC_GC_HL_MC.ordinal()) {
                extractArcCodeWPPP(code, x);        // HL, pGC, GC, MC
                gp = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_nGC_HL_MC.ordinal()) {
                extractArcCodeWPPP(code, x);        // HL, GC, nGC, MC
                gp = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_pHC_HL_MC.ordinal()) {
                extractArcCodeWPPP(code, x);        // HL, GC, pHC, MC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeWP(WORDFV_W0Pp, x[0], x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_HL_nHC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);        // HL, GC, nHC, MC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeWP(WORDFV_W0Pn, x[0], x[2]);
                mod = createWordCodeP(WORDFV_P0, x[3]);
            } else if (temp == GC_HL_pMC_MC.ordinal()) {
                extractArcCodeWPPP(code, x);        // HL, GC, pMC, MC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
            } else if (temp == GC_HL_MC_nMC.ordinal()) {
                extractArcCodeWPPP(code, x);        // HL, GC, MC, nMC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeW(WORDFV_W0, x[0]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
            } else if (temp == pGC_GC_HC_ML.ordinal()) {
                extractArcCodeWPPP(code, x);        // ML, pGC, GC, HC
                gp = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
                head = createWordCodeP(WORDFV_P0, x[3]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == GC_nGC_HC_ML.ordinal()) {
                extractArcCodeWPPP(code, x);        // ML, GC, nGC, HC
                gp = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
                head = createWordCodeP(WORDFV_P0, x[3]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == GC_pHC_HC_ML.ordinal()) {
                extractArcCodeWPPP(code, x);        // ML, GC, pHC, HC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == GC_HC_nHC_ML.ordinal()) {
                extractArcCodeWPPP(code, x);        // ML, GC, HC, nHC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
                mod = createWordCodeW(WORDFV_W0, x[0]);
            } else if (temp == GC_HC_pMC_ML.ordinal()) {
                extractArcCodeWPPP(code, x);        // ML, GC, HC, pMC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeWP(WORDFV_W0Pp, x[0], x[3]);
            } else if (temp == GC_HC_ML_nMC.ordinal()) {
                extractArcCodeWPPP(code, x);        // ML, GC, HC, nMC
                gp = createWordCodeP(WORDFV_P0, x[1]);
                head = createWordCodeP(WORDFV_P0, x[2]);
                mod = createWordCodeWP(WORDFV_W0Pn, x[0], x[3]);
            } else if (temp == GC_HC_MC_pGC_pHC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
                head = createWordCodePP(WORDFV_PpP0, x[4], x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GC_HC_MC_pGC_pMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[4], x[2]);
            } else if (temp == GC_HC_MC_pHC_pMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodePP(WORDFV_PpP0, x[3], x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[4], x[2]);
            } else if (temp == GC_HC_MC_nGC_nHC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
                head = createWordCodePP(WORDFV_P0Pn, x[1], x[4]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GC_HC_MC_nGC_nMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
            } else if (temp == GC_HC_MC_nHC_nMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodePP(WORDFV_P0Pn, x[1], x[3]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
            } else if (temp == GC_HC_MC_pGC_nHC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
                head = createWordCodePP(WORDFV_P0Pn, x[1], x[4]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GC_HC_MC_pGC_nMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
            } else if (temp == GC_HC_MC_pHC_nMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodeP(WORDFV_P0, x[0]);
                head = createWordCodePP(WORDFV_PpP0, x[3], x[1]);
                mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
            } else if (temp == GC_HC_MC_nGC_pHC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
                head = createWordCodePP(WORDFV_PpP0, x[4], x[1]);
                mod = createWordCodeP(WORDFV_P0, x[2]);
            } else if (temp == GC_HC_MC_nGC_pMC.ordinal()) {
                extractArcCodePPPPP(code, x);
                gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
                head = createWordCodeP(WORDFV_P0, x[1]);
                mod = createWordCodePP(WORDFV_PpP0, x[4], x[2]);
            } else {
                continue;
            }

            int headId;
            int modId;
            int gpId = 0;
            int dir = 0;
            int pdir = 0;
            headId = wordAlphabet.lookupIndex(head);
            modId = wordAlphabet.lookupIndex(mod);
            if (gp != -1) {
                gpId = wordAlphabet.lookupIndex(gp);
                if (dist != 0) {
                    dir = ((dist >> 1) & 1) + 1;
                    pdir = ((dist >> 2) & 1) + 1;
                }
            }

            if (headId >= 0 && modId >= 0 && gpId >= 0) {
                int id = hashcode2int(code) & numberLabeledArcFeatures;
                if (id < 0) continue;
                float value = params.getParamsL()[id];
                if (gp == -1) {
                    int[] y = {headId, modId, dist * params.getT() + label};
                    tensor.add(y, value);
                } else {
                    int[] y = {gpId, headId, modId, pdir * params.getT() + plabel, dir * params.getT() + label};
                    tensor2.add(y, value);
                }
            }
        }

    }

}

class LazyCollector implements Collector {

    //This is a dummy collector but a real one is used in LocalFeatureData class (getLabelFeature, getLabelScoreTheta)

    @Override
    public void addEntry(int x) {
        //This is a dummy method, but is call in some methods inside SyntacticFeatureFactory class
    }

    @Override
    public void addEntry(int x, float va) {
        //This is a dummy method, but is call in some methods inside SyntacticFeatureFactory class
    }

}
