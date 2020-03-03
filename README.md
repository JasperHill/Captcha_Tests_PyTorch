Captcha_Test_Pytorch
------

This work investigates the efficacy of a neural network based on the high-rank linear operator as described in `Writeups/Writeup.pdf`. Specifically, the operator is tested within the context of a generative adversarial network (GAN). The objective is to closely replicate Captcha images obtained from the `samples` subdirectory. Should this operator prove successful, the generator will be used to train a Captcha solver.

## Model and hyperparameter initialization
Within the `build_tools` subdirectory, calling `Captcha_GAN.py --save_model` will initialize the GAN and create state dictionaries for the generator and discriminator at `build_tools/gen_saves.pth` and `build_tools/disc_saves.pth` respectively. All configuration details are defined in `MODEL_CONSTANTS.py`. They are:
* `BATCH_SIZE`
* `NUM_EPOCHS`
* `EPSILON`, which is the range for false and true labels (i.e. (0 - `EPSILON`) for false and (1-`EPSILON` - 1) for true)
* `GEN_LR`, which is the generator learning rate
* `DISC_LR`, the discriminator learning rate
* `GEN_SAVE_PATH`, the path for the generator state dictionary
* `DISC_SAVE_PATH`, same as above but for the discriminator
* `IMG_HEIGHT`, `IMG_WIDTH`, and `IMG_CHANNELS`
* `alphanumeric`, the full set of allowed characters in the Captcha strings
* `N`, the number of characters in each string

## Data preprocessing
The dataset object returns several different objects for a given `idx`. The first is a rank-3 floating point tensor of the image. The second is the proper string representation of the image. The final three objects are various numerical representations of the Captcha string. The first, `mat_label`, is essentially a set of 5 one-hot-encoded vectors for each character in the string. They are of dimension 36. `sparse_mat` contains 5 matrices where each one is the outer product of each one-hot vector with itself. Lastly, `dense_mat` is more complicated. It is created by transforming its corresponding Captcha string into a 5-dimensional vector, where each element is the index of `alphanumeric` where the element's corresponding character is located. For example, the string `0a1b2` would map to `[0,10,1,11,2]`. A 5-dimensional square matrix is then created from the outer product of this vector with itself and dividing the matrix by the squared length of `alphanumeric`. It seems that using `sparse_mat` as the generator yields superior performance. A discussion of the mathematical basis for this result may eventually be included in `Writeup.pdf`.

## Training
The training algorithm is executed in `Train_GAN.py`. A running summary of selected network metrics is printed to the screen at the end of each epoch. When the training is complete, a batch of synthetic Captcha images with their intended strings is saved to `synthetic_captchas.pdf`. The GAN state dictionaries are also overwritten.
