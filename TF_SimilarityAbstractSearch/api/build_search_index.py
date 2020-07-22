from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def build_search_index(rels, v):

    # construct a reverse index for suppoorting search
    vocab = v.vocabulary_
    idf = v.idf_
    punc = "'!\"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'" # removed hyphen from string.punctuation
    trans_table = {ord(c): None for c in punc}

    def makedict(s, forceidf=None):
        words = set(s.lower().translate(trans_table).strip().split())
        words = set(w for w in words if len(w) > 1 and (not w in ENGLISH_STOP_WORDS))
        idfd = {}
        for w in words: # todo: if we're using bigrams in vocab then this won't search over them
            if forceidf is None:
                if w in vocab:
                    idfval = idf[vocab[w]] # we have a computed idf for this
                else:
                    idfval = 2.0 # some word we don't know; assume idf 1.0 (low)
            else:
                idfval = forceidf
            idfd[w] = idfval
        return idfd

    def merge_dicts(dlist):
        m = {}
        for d in dlist:
            for k, v in d.items():
                m[k] = m.get(k,1) + v
        return m

    search_dict = []
    for p in rels:
        dict_title = makedict(p['title'], forceidf=11)
        dict_summary = makedict(p['paperAbstract'])
        qdict = merge_dicts([dict_title, dict_summary])
        search_dict.append(qdict)

    return search_dict