
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D,PReLU
from tensorflow.contrib.keras.api.keras.layers import add,Flatten
from generateModel import *

#from generateModel
G = GenPlate("./font/platech.ttf", './font/platechar.ttf', "./NoPlates")

#----------------------------------------------------------------------------------------------
#生成器

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
   # generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):# 生成一个batch_size的数据
            a=G.genPlateString(-1,-1)
            b=G.generate(a)
            img = cv2.resize(b,(224,224))
            X[i] = img.astype('float32')
            for j, ch in enumerate(a):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def decode(y):
    '''
    从预测结果矩阵中取出返回需要的表达字符串
    :param y: 
    :return: 
    '''
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[i] for i in y])


#----------------------------------------------------------------------------------------------
'''
model settings
'''

seed = 10
np.random.seed(seed)





def _BN_ReLU_Conv2d(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        relu_name = name + '_relu'
        conv_name = name + '_conv'

    else:
        bn_name = None
        relu_name = None
        conv_name = None
    x = BatchNormalization(axis=3, name=bn_name)(x)
    x = PReLU()(x)
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation=None, name=conv_name)(x)

    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = _BN_ReLU_Conv2d(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = _BN_ReLU_Conv2d(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = _BN_ReLU_Conv2d(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

characters = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
width, height, n_len, n_class = 224, 224, 7, len(characters)

#ResNet34v2模型
def ResNet34V2_model():
    inpt = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((3, 3))(inpt)
    x = _BN_ReLU_Conv2d(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (56,56,64)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # (28,28,128)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # (14,14,256)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # (7,7,512)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    # x = Dense(1000,activation='softmax')(x)
    x = [Dense(n_class, activation='softmax', name='P%d' % (i + 1))(x) for i in range(7)]
    model = Model(inputs=inpt, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model
model=ResNet34V2_model()
# print(model.summary())

"""
使用者三段代码测试生成结果，以及解码结果
X, y = next(gen(1))
plt.imshow(X[0])
plt.show()
print(X.shape)
print(decode(y))
plt.title(decode(y))
"""
print("train start")
#训练的时候每轮1000个样本共5轮，一个batch_size=32，所以一共有16W张图片
# model.fit_generator(gen(), samples_per_epoch=1000,nb_epoch=5,
#                     nb_worker=1, pickle_safe=True,
#                     validation_data=gen(), nb_val_samples=1280)
model.fit_generator(gen(),steps_per_epoch=100,epochs=5,validation_data=gen(),validation_steps=100,verbose=1)
print("train over")
print("model save")
#保存模型图
from tensorflow.contrib.keras.api.keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes='True')
model.save('resnet34_v2_model.h5')



#测试下结果
X, y = next(gen(1))
y_pred = model.predict(X)
print(X)
print("test start")
print(X[0])
print(decode(y))
print(decode(y_pred))
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
# plt.imshow(X[0], cmap='gray')
plt.imshow(X[0])
plt.show()
#这里显示是和正常的颜色不一样，这是因为，plt读取的通道顺序和cv2的通道顺序是不同的

score=model.evaluate(X,y,verbose=0)
print("test score=",score[0])
print("test accuracy=",score[1])