class BaseConfig:
    trainPath = 'resource/train-augment-wholecontent.txt'
    valPath = 'resource/val-init.txt'
    testPath = 'resource/test-init-alter-5.txt'
    w2vModel = 'resource/word_embedding.json'
    w2vModel_ex = 'resource/word_embedding_extend.json'  #这个是加入标点符号的embedding
    rf_model_path = 'result/model/RandomForest/rf_rm2json-dict30bool-rules-v2.pkl'
    rf_dict_path = 'resource/lawdict.txt'
    stpPath = 'resource/stopwords.txt'
    lawKsPath = 'resource/ft_ks.json'
    lawQHJPath = 'resource/law_qhj_dict.json'

    char_vocab_size = 4594  # *
    word_vocab_size = 97505
    char_dimension = 30
    word_dimension = 128
    law_word_dimension = 128

    # training
    batch_size = 64  # alias = N
    lr = 1e-3  #
    logdir = 'logdir'  # log directory
    num_epochs = 200
    save_per_batch = 100
    print_per_batch = 10

class BasicConfig2(BaseConfig):
    trainPath = 'resource/gyshz_traindata/train-init.txt'
    valPath = 'resource/gyshz_traindata/val-init.txt'
    testPath = 'resource/gyshz_traindata/test-init.txt'
    w2vModel = 'resource/gyshz_traindata/word2vec/word_embedding.json'






