'''
Extract topics from
'''

import pyLDAvis.gensim
from corpus import Corpus
import sys
from gensim.models import LdaModel, LdaMulticore
import logging

if __name__ == '__main__':
    posts_file = sys.argv[1]
    dict_file = sys.argv[2]
    model_save = sys.argv[3]
    n_topics = int(sys.argv[4])
    passes = int(sys.argv[5])
    prefix = sys.argv[6]

    logging.basicConfig(
        filename=prefix + 'lda_model' + str(n_topics) + '_' + str(passes) + '.log',
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
    )
    
    corpus = Corpus(posts_file, dict_file)

    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=corpus.dictionary,
        random_state=100,
        num_topics=n_topics,
        passes=passes,
        chunksize=1000,
        batch=False,
        alpha='asymmetric',
        decay=0.5,
        offset=64,
        eta=None,
        eval_every=0,
        iterations=100,
        gamma_threshold=0.001,
        per_word_topics=True
    )

    lda_model.save(model_save)
    corpus = Corpus(posts_file, dict_file)
    
    visualization = pyLDAvis.gensim.prepare(lda_model, corpus, corpus.dictionary)

    dest = prefix + '_vis.html'
    with open(dest, 'w') as handle:
        pyLDAvis.save_html(visualization, handle)
