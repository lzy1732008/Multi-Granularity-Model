class BaseConfig:
    trainPath = 'resource/train-init.txt'
    valPath = 'resource/val-init.txt'
    testPath = 'resource/test-init.txt'
    w2vModel = 'resource/word_embedding.json'
    w2vModel_ex = 'resource/word_embedding_extend.json'
    char_vocab_size = 4594  # *
    word_vocab_size = 97505
    char_dimension = 30
    word_dimension = 128
    law_word_dimension = 128

    # training
    batch_size = 64  # alias = N
    lr = 4e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory
    num_epochs = 200
    save_per_batch = 100
    print_per_batch = 10




