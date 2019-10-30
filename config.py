# paths
qa_path                    = 'data'                # directory containing the question and annotation jsons
bottom_up_trainval_path    = 'data/trainval_36'    # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path        = 'data/test2015_36'    # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved
preprocessed_test_path     = 'genome-test.h5'      # path where preprocessed features from the test split are saved
vocabulary_path            = 'vocab.json'          # path where the used vocabularies for question and answers are saved

task    = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# preprocess config
output_size         = 36     # max number of object proposals per image
max_question_length = 0     # max question length. Set to 0 if you take the largest question length of the train dataset

# gpu
gpuid   = '1'

# training config
epochs       = 100
batch_size   = 256
initial_lr   = 2.0e-3
lr_halflife  = 50000     # in iterations
data_workers = 4

# =======================================
# model parameters
# =======================================
iteration   = 2

# image
output_features             = 2048
hidden_features             = 512
spatial_features            = 5
interIntraBlocks_dropout    = 0.1
num_inter_head              = 8
num_intra_head              = 8
num_block                   = 2
spa_block                   = 2
que_block                   = 2
visual_normalization        = True

# question
question_features       = 1280
embedding_features      = 300
embedding_dropout       = 0.1

# classifier
mid_features        = 1024
max_answers         = 3000
classifier_dropout  = 0.1