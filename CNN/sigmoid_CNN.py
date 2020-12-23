from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Input, Conv2DTranspose, UpSampling2D, \
    BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import initializers
from skimage import io
import cv2
import numpy as np
import math
import os

'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
'''
# Xavier initialization
initializers = initializers.glorot_normal()  # or glorot_uniform()

num_smaple = 7900  # total 201
# width = 256
# height = 256
width = 600
height = 600
channel = 1
b_size = 2
epoc = 10
step_per_epoch = math.ceil(num_smaple / b_size)
path = 'D:\pycharm project'


# 1.Prepare DATA
def train_generator(path, b_size):

    #ori_files = os.listdir('D:\\training_set\src') #"D:\FYP\original"
    ori_files = os.listdir('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\original')  # "D:\FYP\original"
    # gt_path = os.listdir("D:\FYP\\truth")  # dif_truth dinary_truth_600
    # save_path='D:\pycharm project\softmax01\\'
    while True:
        i = 0
        a = np.zeros((b_size, width, height,1))
        b = np.zeros((b_size, width, height))
        while i < b_size:
            # The two orresponding images have the same name saved in two folders
            e=np.zeros((width,height,1))
            #c = np.array(io.imread( 'D:\\training_set\src\\'+ ori_files[i])) #"D:\FYP\original\\"
            c = np.array(io.imread('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\original\\' + ori_files[i]))  # "D:\FYP\original\\"
            c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            e[:,:,0]=c
            #d = np.array(io.imread('D:\\training_set\label\\'+ ori_files[i], as_gray=True)) # "D:\FYP\\truth\\"
            d = np.array(io.imread('D:\珩珩工作室\polyu study\EIE4433\戴宪洁大佬\CNN训练资源\\truth\\' + ori_files[i], as_gray=True))  # "D:\FYP\\truth\\"
            a[i] = e
            b[i] = d
            i = i + 1
        yield (a / 255, b / 255)


def binary_operation(in_matrix):
    # one = tf.ones_like((600, 600),dtype=float)  # 生成与a大小一致的值全部为1的矩阵
    # zero = tf.zeros_like((600, 600),dtype=float)
    # label = tf.where(in_matrix[:, :, 0] < 0.5, x=0.0, y=1.0)
    return in_matrix[:, :, :, 0]


# 2.CNN structure
# ENCODER
encoder_input = Input(shape=(width, height,1))

first_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers)(encoder_input)

second_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                       kernel_initializer=initializers)(first_output)

third_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers)(second_output)
'''
third11_output = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                        kernel_initializer=initializers)(second_output)

merge_two = Concatenate()([first_output, second_output, third_output, third11_output])
'''
merge_two = Concatenate()([first_output, second_output, third_output])
forth_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation=None,
                      kernel_initializer=initializers)(merge_two)
'''
first_output1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers)(encoder_input)
first_output2=BatchNormalization()(first_output1)
first_output=Activation('relu')(first_output2)

second_output1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers)(first_output)
second_output2=BatchNormalization()(second_output1)
second_output=Activation('relu')(second_output2)

third_output1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers)(second_output)
third_output2=BatchNormalization()(third_output1)
third_output=Activation('relu')(third_output2)

merge_two = Concatenate()([first_output, second_output, third_output])

forth_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation=None,
                      kernel_initializer=initializers)(merge_two)
'''
encoder_output = MaxPool2D(pool_size=(2, 2))(forth_output)

encoder = Model(inputs=encoder_input, outputs=encoder_output, name='ENCODER')
# encoder.summary()
#plot_model(encoder, "encoder.png", show_shapes=True)

# DECODER
# decoder_input = Input(shape=(128, 128, 64))
decoder_input = Input(shape=(300, 300, 64))
fifth_output = UpSampling2D(size=(2, 2))(decoder_input)
sixth_output = Conv2DTranspose(32, kernel_size=(3, 3), activation="relu", padding='same')(fifth_output)
seventh_ouput = Conv2DTranspose(32, kernel_size=(3, 3), activation="relu", padding='same')(sixth_output)
decoder_output = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid',
                        kernel_initializer=initializers)(seventh_ouput)
# decoder_output

decoder = Model(inputs=decoder_input, outputs=decoder_output, name='DECODER')
# decoder.summary()
#plot_model(decoder, "decoder.png", show_shapes=True)

# pixel-wise detection CNN
input_img = keras.Input(shape=(width, height,1), name="img")
encoded_img = encoder(input_img)
decoded_im = decoder(encoded_img)
# decoded_img = Lambda(binary_operation)(decoded_im)

pixel_detection = keras.Model(input_img, decoded_im, name="autoencoder")
# pixel_detection.summary()

# 3.TRAIN the model keras.optimizers.Adadelta(rho=0.9)  keras.optimizers.SGD(learning_rate=0.01,momentum=0.5)
pixel_detection.compile(optimizer=keras.optimizers.Adadelta(rho=0.9),
                        loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
'''
#gt_path = os.listdir("D:\FYP\\truth")
#gg = np.array(io.imread("D:\FYP\\truth\\" + gt_path[0], as_gray=True))
#weights = class_weight.compute_class_weight('balanced', np.unique(gg), gg)
#print(weights)
'''

# train=pixel_detection.fit(x_train, y_train, epochs=epoc, batch_size=b_size)
train = pixel_detection.fit_generator(generator=train_generator(path, b_size), steps_per_epoch=step_per_epoch,
                                      epochs=epoc, verbose=2)  # ,class_weight=weights)
#pixel_detection.save('mymodel4_30')
pixel_detection.save('mymodel4_0.985')
