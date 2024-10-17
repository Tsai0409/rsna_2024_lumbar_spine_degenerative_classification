def reduce_mem_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


### time fs ###
import pandas as pd
def extract_time_fs(df, col):
    df[col] = pd.to_datetime(df[col])
    df['year'] = df[col].dt.year
    df['month'] = df[col].dt.month
    df['day'] = df[col].dt.day
    df['hour'] = df[col].dt.hour
    df['minute'] = df[col].dt.minute
    return df

# col = 'time'
# train = extract_time_fs(train, col)
# train[['year', 'month', 'day', 'hour', 'minute']]
### time fs ###


### word2vec ###
import gensim
from gensim.models import KeyedVectors
def get_sentence_vector(x: str, ndim=300):
    embeddings = [
        w2v_model.get_vector(word)
        if word in w2v_model
        else np.zeros(ndim, dtype=np.float32)
        for word in x.split()
    ]
    if len(embeddings) == 0:
        return np.zeros(ndim, dtype=np.float32)
    else:
        return np.mean(embeddings, axis=0)

def get_word2vec_fs(df, col, ndim):
    return np.stack(
        train[col].fillna("").str.replace("\n", "").map(
            lambda x: get_sentence_vector(x)
        ).values
    )
# word2vec_path = '/groups/gca50041/ariyasu/word2vec/GoogleNews-vectors-negative300.bin'
# w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary = True)

# ndim = 300
# col = 'comment_1'

# word2vec_fs = get_word2vec_fs(train, col=col, ndim=ndim)
# train[[f'{col}_w2v_{i}' for i in range(ndim)]] = word2vec_fs
### word2vec ###

### tfidf ###
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation

def tfidf_vectorize(X, max_features=100, stop_words='english', svd=False):
    if svd:
        tfidf_max_feature = max_features*100
    else:
        tfidf_max_feature = max_features

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words=stop_words,
        ngram_range=(1, 3),
        max_features=tfidf_max_feature)
    word_vectorizer.fit(X)
    word_features = word_vectorizer.transform(X)
    word_features = np.array(word_features.toarray())

    if svd:
        print('svd...')
        svd = TruncatedSVD(n_components=max_features, random_state=42)
        word_features = svd.fit_transform(word_features)

    return word_features
    # return word_vectorizer, word_features

def tfidf_vectorize_with_char(X, max_features=100, stop_words='english', svd=False):
    if svd:
        tfidf_max_feature = max_features*100
    else:
        tfidf_max_feature = max_features

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words=stop_words,
        ngram_range=(2, 5),
        max_features=tfidf_max_feature)
    char_vectorizer.fit(X)

    char_features = char_vectorizer.transform(X)
    char_features = np.array(char_features.toarray())

    if svd:
        print('svd...')
        svd = TruncatedSVD(n_components=max_features, random_state=42)
        char_features = svd.fit_transform(char_features)

    return char_features

# df = pd.concat([train, test])

# col = 'comment_1'
# stop_words = 'english'

# max_features = 100
# tfidf_cols = [f'tfidf_{col}_{i}' for i in range(max_features)]
# df[tfidf_cols] = tfidf_vectorize(df[col].fillna(''), max_features=max_features, stop_words=stop_words, svd=True)

# max_features_char = 100
# tfidf_char_cols = [f'tfidf_char_{col}_{i}' for i in range(max_features)]
# df[tfidf_char_cols] = tfidf_vectorize_with_char(df[col].fillna(''), max_features=max_features_char, stop_words=stop_words, svd=True)


# train = df.iloc[:len(train)]
# test = df.iloc[len(train):]
### tfidf ###

### fasttext ###
import fasttext
# import fasttext.util
# fasttext.util.download_model('ja', if_exists='ignore')
# ft_model = fasttext.load_model('/groups/gca50041/ariyasu/fasttext/cc.en.300.bin') # for english

# col = 'comment_1'

# fasttext_features = np.stack(train[col].fillna("").map(
#         lambda x: ft_model.get_sentence_vector(x)
#     ).values
# )
# fasttext_cols = [f'fasttext_{col}_{i}' for i in range(300)]
# train[fasttext_cols] = fasttext_features
### fasttext ###

###  ###
###  ###

