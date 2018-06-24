from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd


def load_articles(filename, num_chunks=None):
    print("loading articles...")
    chunksize = 10 ** 6
    articles_list = []
    i = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        if num_chunks is None or i < num_chunks:
            articles_list.append(chunk)
            i += 1
        else:
            break
    articles = pd.concat(articles_list)
    return articles


def load_comments(filename, num_chunks=None):
    print("loading comments...")
    chunksize = 10 ** 6
    comments_list = []
    i = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        if num_chunks is None or i < num_chunks:
            comments_list.append(chunk)
            i += 1
        else:
            break
        comments = pd.concat(comments_list)
    return comments


def train_doc2vec(data):
    print("training")
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")


def comment_belongs2article(model, articles, comments, comment_index):
    sample_comment = comments["article_id"][comment_index]
    article_id = [x for x in articles["article_id"] if x == sample_comment][0]
    index = 0
    for articles_id in articles["article_id"]:
        if articles_id == article_id:
            break
        index += 1

    comment_tokenized = word_tokenize(comments["comment_text"][comment_index].lower())
    article_tokenized = word_tokenize(articles["article_text"][index].lower())
    # train_doc2vec(articles["article_text"])
    comment_vector = model.infer_vector(comment_tokenized)
    article_vector = model.infer_vector(article_tokenized)

    # to find most similar doc using tags

    similar_doc = model.docvecs.most_similar(positive=[comment_vector])

    for article in similar_doc:
        if int(article[0]) == index:
            return True

    return False


def main():
    model = Doc2Vec.load("d2v.model")
    articles = load_articles("../SOCC/raw/gnm_articles.csv")
    comments = load_comments("../SOCC/raw/gnm_comments.csv")
    comment_index = 1000
    print("belongs to ", comment_belongs2article(model, articles, comments, comment_index))


if __name__ == "__main__":
    main()
