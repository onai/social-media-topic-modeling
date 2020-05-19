'''
Produce visualizations of LDA models without
having to spin up a notebook
'''
from corpus import Corpus
from gensim import corpora, models
from pprint import pprint
import pyLDAvis.gensim
import sys

if __name__ == '__main__':
    model_file = sys.argv[1]
    corpus_file = sys.argv[2]
    dict_file = sys.argv[3]
    dest = sys.argv[4]

    the_corpus = Corpus(corpus_file, dict_file)
    viz_dest = dest + '.vis.html'

    lda = models.LdaModel.load(model_file)

    # first save visualization
    visualization = pyLDAvis.gensim.prepare(lda, the_corpus, the_corpus.dictionary)
    with open(viz_dest, 'w') as handle:
        pyLDAvis.save_html(visualization, handle)

    # dump topics:
    with open(dest, 'w') as handle:
        sys.stdout = handle
        pprint(lda.print_topics())
