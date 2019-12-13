class BaseConfig:
    trainPath = 'resource/train-init.txt'
    valPath = 'resource/val-init.txt'
    testPath = 'resource/test-init.txt'
    w2vModel = 'resource/word_embedding.json'
    w2vModel_ex = 'resource/word_embedding_extend.json'  #这个是加入标点符号的embedding
    rf_model_path = 'result/model/RandomForest/rf_rm2json-dict30bool-rules-v2.pkl'
    rf_dict_path = 'resource/lawdict.txt'
    stpPath = 'resource/stopwords.txt'

    char_vocab_size = 4594  # *
    word_vocab_size = 97505
    char_dimension = 30
    word_dimension = 128
    law_word_dimension = 128

    # training
    batch_size = 64  # alias = N
    lr = 4e-5  #
    logdir = 'logdir'  # log directory
    num_epochs = 200
    save_per_batch = 100
    print_per_batch = 10




