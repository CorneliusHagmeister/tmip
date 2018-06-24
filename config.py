class Config:

    # pre processing params
    padding_len=300
    # training params
    batch_size = 256
    num_epochs = 9
    load_model=False
    load_model_path="./models/first_running.h5"
    save_model=True
    save_model_path="./models/first_running.h5"

    # model paramters
    num_filters = 64
    embed_dim = 50
    weight_decay = 1e-4
    num_classes = 2
    lr=0.0001
    load_embedding_weights=True
    save_embedding_weights=False