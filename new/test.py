#%%
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras 
import imageio
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import multi_gpu_model
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50


#%%

def transfer_model(X_train,y_train,batch_size,epoch):
    base_model = ResNet50(include_top=False, weights='imagenet',input_shape=(224,224,3))
    for layers in base_model.layers:
        layers.trainable = False
    model = Flatten()(base_model.output)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(121, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=model)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
    
    callbacks = [
    # 把TensorBoard日志写到'logs'
    keras.callbacks.TensorBoard(log_dir='./logs'),
    # 当categorical_accuracy，也就是分类精度在10个epoh之内都没提升时，降低learning rate
    keras.callbacks.ReduceLROnPlateau(monitor='categorical_accuracy', patience=10, verbose=2),
    # 当categorical_accuracy在15个epoch内没有提升的时候，停止训练
    keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=15, verbose=2)]
    
    model.fit(X_train,y_train, batch_size=batch_size, epochs=epoch)
    # parallel_model = multi_gpu_model(model, gpus=2)
    # parallel_model.compile(loss='categorical_crossentropy',
    #                    optimizer='adam',metrics=['accuracy']) 
    # parallel_model.fit(X_train,y_train, batch_size=batch_size, epochs=epoch)
    return model

def get_filenames(path_in):
    files = []
    filenames = []
    f_list = os.listdir(path_in)
    # print f_list
    for i in f_list:
        # os.path_in.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.jpg':
            files.append(os.path.splitext(i)[0])
    files=sorted(files,key=int)
    for i in files:
        i = (i+'.jpg')
        filenames.append(i)
    return filenames

def preparing_data(path_in,path_out):
    filenames = get_filenames(path_in)

    total = 0
    for image in filenames:
        img_0 = cv2.imread('./data'+'/'+image)
        img_1 = cv2.resize(img_0, (224, 224))
        img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        total += 1
        cv2.imwrite(path_out+'/'+str(total)+'.jpg',img_2)
        total += 1
        cv2.imwrite(path_out+'/'+str(total)+'.jpg',img_2)
        total += 1
        cv2.imwrite(path_out+'/'+str(total)+'.jpg',img_2)
        total += 1
        cv2.imwrite(path_out+'/'+str(total)+'.jpg',img_2)
        total += 1
        cv2.imwrite(path_out+'/'+str(total)+'.jpg',img_2)

def preparing_label(path_in):
    filenames = get_filenames(path_in)

    counter = 0 
    count=0
    labels = pd.DataFrame()
    for filename in filenames:
        if count == 0:
            labels_data = pd.DataFrame({'image': filename, 'labels': str(counter)},index=[count])
            labels = pd.concat((labels, labels_data))
            count+=1

        elif count % 5 == 0:
            counter+=1
            labels_data = pd.DataFrame({'image': filename, 'labels': str(counter)},index=[count])
            labels = pd.concat((labels, labels_data))
            count+=1
        else:
            labels_data = pd.DataFrame({'image': filename, 'labels': str(counter)},index=[count])
            labels = pd.concat((labels, labels_data))
            count+=1
    return labels

def imageToarray(labels, path_in):
    train=[]
    for filename in labels.image:
        train.append(np.array(Image.open(path_in+'/'+filename)))
    train_data = np.array(train)
    #shape(605, 224, 224, 3)
    return train_data

def one_hotprocess(labels):
    lbl = LabelEncoder().fit(list(labels['labels'].values))
    labels['code_labels'] = pd.DataFrame(lbl.transform(list(labels['labels'].values)))
    return labels.code_labels

path_in='./data'
path_out='./resizeddata'

#labels = preparing_data(path_in,path_out)
labels = preparing_label(path_out)
X = imageToarray(labels,path_out)
y = one_hotprocess(labels)
#np.save('train.npy', X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#将数据类型设为‘float32’，并归一化到0~1区间
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train,121)
y_test = np_utils.to_categorical(y_test,121)

#start train
epoch = 50
batch_size = 256
model = transfer_model(X_train,y_train,batch_size,epoch)
model.save('speed.h5')

#detection
model= load_model('speed.h5')

img_path = './resizeddata/6.jpg'
img = image.load_img(img_path,target_size=(224,224))

x = image.img_to_array(img)* 1.0 / 255
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# result_list=[]
# for i in preds[0]:
#     result_list.append(i)
# speed = result_list.index(max(result_list))
# speed
# %%
img = cv2.imread(img_path)
cv2.imshow('image', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
