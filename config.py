# Minibatch Size
BATCH_SIZE = 128
# Learning rate
LEARNING_RATE = 0.01
# Number of epochs for training
NUM_EPOCHS = 30

# dataset params
def get_train_config(dataset='amazon_review'):
    return configs[dataset]

adv_hp = {"nb_epoch": 20, "batch_size": 256, "lr": 1e-3, "k": 5, "l": 1, "hp_lambda": 1e-7, "plt_frq": 5, "dropout_rate": 0.2, "adv_disc_depth": 1, "adv_disc_hidden_dim": 500}

nn_hp = [{"dropout_rate": 0.2, "filter_length_list": [3, 4, 5], "nb_filter": 200, "l2_constraint": 10, "use_pretrained_embedding": True}, {"dropout_rate": 0.2, "filter_length_list": [3, 4, 5], "nb_filter": 200, "l2_constraint": 1000, "use_pretrained_embedding": True}]



configs = {

            'amazon_review':
                {
                    'batch_size':256,
                    'lr':0.001,
                    'nb_epoch':30,
                    'label_type':'multi-class',
                    'temp':3,
                    'adv':False,
                    'adv_hp':adv_hp,
                    'nn_hp':nn_hp,
                },
            'amazon_review_adv':
                {
                    'batch_size':256,
                    'lr':0.001,
                    'nb_epoch':30,
                    'label_type':'multi-class',
                    'temp':3,
                    'adv':True,
                    'adv_hp':adv_hp,
                    'nn_hp':nn_hp,
                },
            }

