import string

BATCH_SIZE =  10
NUM_EPOCHS =  10
EPSILON    = 0.2
GEN_LR     = 1e-2
DISC_LR    = 1e-2

GEN_SAVE_PATH = "./build_tools/gen_saves.pth"
DISC_SAVE_PATH = "./build_tools/disc_saves.pth"

IMG_HEIGHT   =  50
IMG_WIDTH    = 200
IMG_CHANNELS =   3

# the full set of allowed characters in the captcha strings
alphanumeric = string.digits + string.ascii_lowercase

# N == number of characters in each captcha string
N = 5

