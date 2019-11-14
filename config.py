''' Configuration File.
'''

##
# Learning Loss for Active Learning
NUM_TRAIN = 50000 # N
NUM_VAL   = 1000
BATCH     = 64 # B

EPOCH = 300
LR = 0.1
MILESTONES = [150, 225]

MOMENTUM = 0.9
WDECAY = 5e-4