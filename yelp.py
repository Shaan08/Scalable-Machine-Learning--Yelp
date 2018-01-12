#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  27 00:44:01 2017

@author: prashant
"""

#Do the necessary imports
import os
import tensorflow
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from collections import Counter
from scipy.misc import imread
from sklearn.metrics import f1_score
from sklearn.cross_validation import cross_val_score
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.models import Sequential
from keras import regularizers

# ***************Exploring Data******************

#Extracting Labelled Data from training and test dataset.
df = pd.read_csv('/User/prashant/Drive/train.csv') 
photo_to_biz = pd.read_csv('/User/prashant/Drive/train_photo_to_biz_ids.csv')
test_photo_to_biz = pd.read_csv('/User/prashant/Drive/test_photo_to_biz.csv')
train_photos = []
for filename in os.listdir('/User/prashant/Drive/train_photos'):
    val1 = (filename.replace(".jpg",""))#Read the photos and save them as .jpg.
    train_photos.append(int(val1))    
    #print (len(train_photos)) #7458
test_photos = []
for filename in os.listdir('User/prashant/Drive/test_photos'):
    val2 = (filename.replace(".jpg",""))
    test_photos.append(int(val2))
    #print (len(test_photos)) #8303
    print(df.head()) # viewing the data structure, label is given as string.
#    	business_id	       labels
#0	      1000	         1 2 3 4 5 6 7
#1	      1001	         0 1 6 8
#2	      100	             1 2 4 5 6 7
#3	      1006	        1 2 4 5 6

 # Note: the number of pictures in the dataset is variable and there are missing values in the data as well.
 
#Dimension of the train photos.
val3 = train_photos[0:5]
for i in val3:
    img = imread('/User/prashant/Drive/train_photos/' + str(i) + '.jpg')
    print ("Shape:", img.shape)
    print ("Flattened:",img.flatten()) #Eg: Shape: (500, 375, 3),Flattened: [164 122  84 ..., 113  61  22]
    
# There was some discrepancy in the training and test data id which is why we need to only filter out the data which is the dataset.
def get_photo_id_train(biz_id):
    return (list(photo_to_biz.loc[photo_to_biz['business_id'] == biz_id]['photo_id']))
#get_photo_id_train(2)
df_train_pid = photo_to_biz[photo_to_biz['photo_id'].isin(train_photos)]
df_test_id = test_photo_to_biz[test_photo_to_biz['photo_id'].isin(test_photos)]
df_train_pid = df_train_pid.sort_values('photo_id')
reduced_test_p_biz = df_test_id.loc[~df_test_id['photo_id'].duplicated()] # Remove redundant uploads of pictures in Test Set
reduced_test_p_biz = reduced_test_p_biz.sort_values('photo_id')
#reduced_test_p_biz.head()
#Here we get the business Id in alphanumeric format so we added the previously
#acquired jpg appended data as new column to verify the right business ID.
df_train_pid['jpg_num'] = train_photos
reduced_test_p_biz['jpg_num'] = test_photos


# Arranging data in a structure format.
#Take the training data and put it in a dataframe with unique Biz_id and labels assigned to it.
df_train_pid = df_train_pid.sort_values('business_id')
uniq_biz = df_train_pid.sort_values('business_id')
uniq_biz = list(uniq_biz.business_id.unique())
uniq_biz = df[df['business_id'].isin(uniq_biz)]
uniq_biz = uniq_biz.sort_values('business_id')
#uniq_biz.head()

# Now, create a new df with filtered,sorted and unique dataset and then join it with picture id and jpg numbers.
new_df = []
for i in df_train_pid['business_id']:
    if i in uniq_biz['business_id'].values:
        j = uniq_biz[uniq_biz['business_id']==i]
        new_df.append(j)
    else:
        pass    
new_df = pd.concat(new_df)
new_df = new_df.sort_values('business_id')
df_train_pid['labels'] = list(new_df['labels'][0:])
#df_train_pid.head()
one_hot = df_train_pid
one_hot = one_hot.dropna()
t = one_hot['labels']
testing = t[:]


#let us define a function for converting our string label into binary format.like a one hot encoding.
# there is definitely a better way to do it but it works so didn't focus on code iptimization.
def mod_labels(labels):
    #labels in the dataset.
    good_for_lunch = []
    good_for_dinner = []
    takes_reservations = []
    outdoor_seating = []
    restaurant_is_expensive = []
    has_alcohol = []
    has_table_service = []
    ambience_is_classy = []
    good_for_kids = []

    for i in labels:
        vals = i.split() #split the string labels.

        if '0' in vals:
            good_for_lunch.append(1)
        else:
            good_for_lunch.append(0)
            
        if '1' in vals:
            good_for_dinner.append(1)
        else:
            good_for_dinner.append(0)
            
        if '2' in vals:
            takes_reservations.append(1)
        else:
            takes_reservations.append(0)
            
        if '3' in vals:
            outdoor_seating.append(1)
        else:
            outdoor_seating.append(0)
            
        if '4' in vals:
            restaurant_is_expensive.append(1)
        else:
            restaurant_is_expensive.append(0)
            
        if '5' in vals:
            has_alcohol.append(1)
        else:
            has_alcohol.append(0)
            
        if '6' in vals:
            has_table_service.append(1)
        else:
            has_table_service.append(0)
            
        if '7' in vals:
            ambience_is_classy.append(1)
        else:
            ambience_is_classy.append(0)
            
        if '8' in vals:
            good_for_kids.append(1)
        else:
            good_for_kids.append(0)

        if 'nan' in vals:
            pass
        else:
            pass
        
    return good_for_lunch,good_for_dinner,takes_reservations,outdoor_seating ,restaurant_is_expensive ,has_alcohol,has_table_service ,ambience_is_classy ,good_for_kids 

good_for_lunch,good_for_dinner,takes_reservations,outdoor_seating ,restaurant_is_expensive ,has_alcohol,has_table_service ,ambience_is_classy ,good_for_kids = mod_labels(testing)


#Delete the labels with string values and replace them with OHE.

one_hot['good_for_lunch'] = good_for_lunch
one_hot['good_for_dinner'] = good_for_dinner
one_hot['takes_reservations'] = takes_reservations
one_hot['outdoor_seating'] = outdoor_seating
one_hot['restaurant_is_expensive'] = restaurant_is_expensive
one_hot['has_alcohol'] = has_alcohol
one_hot['has_table_service'] = has_table_service
one_hot['ambience_is_classy'] = ambience_is_classy
one_hot['good_for_kids'] = good_for_kids
sorted_one_hot = one_hot.sort_values('jpg_num')
#reduced_test_p_biz.head()
#sorted_one_hot.head()
#Now we have training data in more clean and workable format with 0 or 1 value for presence of labels associated with each biz_id and photo_id.

#We can check to see the range of photo dimension for all images.

train_group = sorted_one_hot[:100]
test_group = reduced_test_p_biz[:100]

def train_dim(df):
    height_d = Counter ()
    width_d = Counter()
    
    for i in df['jpg_num']:
        img = imread('/User/prashant/Drive/train_photos/' + str(i) + '.jpg')
        img = imread('/User/prashant/Drive/test_photos/' + str(i) + '.jpg')
        height,width,depth = img.shape 
        height_d[height] += 1
        width_d[width] += 1
    
    return (height_d),(width_d)

train_height, train_width = train_dim(train_group)
test_height, test_width = train_dim(test_group)



#***************************** CNN Data Preprocessing**************************
# Training Data
resized_train = []
for x in sorted_one_hot[:7000]['jpg_num']:
    img = load_img('/User/prashant/Drive/train_photos/' + str(x) + '.jpg', target_size=(281,281))
    i = img_to_array(img)
    p = np.asarray(i)
    p = p / np.amax(p) 
    resized_train.append(p)
resized_train = np.asarray(resized_train)
resized_train = np.reshape(resized_train,(7000,281,281,3))

#Test Data
resized_test = []
for y in reduced_test_p_biz[:2881]['jpg_num']:
    img = load_img('/User/prashant/Drive/test_photos/' + str(y) + '.jpg', target_size=(281,281))
    j = img_to_array(img)
    q = np.asarray(j)
    q = q / np.amax(q) 
    resized_test.append(q)

resized_test = np.asarray(resized_test)
resized_test = np.reshape(resized_test,(2881,281,281,3))
#print (resized_train.shape) 
#print (resized_test.shape)
multi_train_labels = np.asarray(sorted_one_hot[['has_table_service','has_alcohol','good_for_kids','takes_reservations','outdoor_seating','good_for_dinner','good_for_lunch','ambience_is_classy', 'restaurant_is_expensive']])
#print ("Train_Labels:", '\n', multi_train_labels[:5])
# Splitting our training data into train,test and validation
X1_train, X1_test, y1_train, y1_test = train_test_split(resized_train, multi_train_labels[:7000], test_size=0.3)

#Weighing class to fix variable distribution
#https://github.com/fchollet/keras/issues/5116
one = sum(sorted_one_hot['has_table_service'])
two = sum(sorted_one_hot['has_alcohol']) 
three = sum(sorted_one_hot['good_for_kids'])
four = sum(sorted_one_hot['takes_reservations']) 
five = sum(sorted_one_hot['outdoor_seating'])
six = sum(sorted_one_hot['good_for_dinner'])
seven = sum(sorted_one_hot['good_for_lunch']) 
eight = sum(sorted_one_hot['ambience_is_classy'])
nine = sum(sorted_one_hot['restaurant_is_expensive'])
my_dictionary = {0:one, 1:two, 2: three, 3: four, 4: five, 5:six, 6:seven, 7:eight, 8:nine}
def get_weights(dictionary,smooth_factor):
    if smooth_factor > 0:
        p = max(my_dictionary.values()) * smooth_factor
        for k in my_dictionary.keys():
            my_dictionary[k] += p
    majority = max(my_dictionary.values())
    return {cls: float(majority / count) for cls, count in my_dictionary.items()}
class_weights = get_weights(my_dictionary,1)
class_weights


# ******************** Beginning with CNN 1 layer architecture*************************
model = Sequential()
model.add(Convolution2D(input_shape=(281,281,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", name='convolution2d'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=9, activation='sigmoid', input_shape=[636192], kernel_regularizer=regularizers.l2(.01)))
#model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history1 = model.fit(X1_train,y1_train, validation_split=.1, class_weight=class_weights)

p1 = model.predict(X1_test)
scores(p1,y1_test) #('F1 Score:',0.65,'Precision:',0.68,'Recall:',0.63)
model.evaluate(X1_test,y1_test) # [0.6595032240095593, 0.64962963376726424]


#********************* Using CNN 2 layers *******************************************

model = Sequential()
model.add(Convolution2D(input_shape=(281,281,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='first_convolution1'))
model.add(MaxPooling2D(pool_size=(2,2),strides=1, padding='same'))
model.add(Convolution2D(input_shape=(281,281,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='second_convolution'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=9, activation='sigmoid', input_shape=[636192], kernel_regularizer=regularizers.l2(.01)))
#model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history2 = model.fit(X1_train,y1_train, validation_split=.1, epochs=10, class_weight=class_weights)
p2 = model.predict(X1_test)
scores(p2,y1_test) #('F1 Score:',0.68,'Precision:',0.65,'Recall:',0.71)
model.evaluate(X1_test,y1_test) #[0.67540292217617948, 0.65037037259056452]

#Since we are getting a score of 68%, I wanted to try VGG net architecture on my data
#Increasing epochs got a better accuracy and loss was lowered but F1 score remained pretty much similar.
model = Sequential()
model.add(Convolution2D(input_shape=(281,281,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='first_convolution1'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))

model.add(Convolution2D(input_shape=(141,141,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='second_convolution'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))

model.add(Convolution2D(input_shape=(71,71,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='third_convolution'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))

model.add(Convolution2D(input_shape=(36,36,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='fourth_convolution'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))

model.add(Convolution2D(input_shape=(18,18,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='fifth_convolution'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))

model.add(Convolution2D(input_shape=(5,5,3), filters=32, kernel_size=3, activation='relu', padding='same', data_format="channels_last", 
                      name='sixth_convolution'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=9, activation='sigmoid', input_shape=[800], kernel_regularizer=regularizers.l2(.01)))
#model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history3 = model.fit(X1_train,y1_train, validation_split=.1, class_weight=class_weights)
p3 = model.predict(X1_test)
scores(p3,y1_test) #('F1 Score:',0.69,'Precision:',0.69,'Recall:',0.68)
model.evaluate(X1_test,y1_test) # [0.60740917864299959, 0.67455026717413036]



#******************* VGG NET***************************************
#model = Sequential()
#model.add(keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(281,281,3)))
#model.add(Flatten())
#model.add(Dense(units=9, activation='sigmoid', input_shape=[32768], kernel_regularizer=regularizers.l2(.01)))
#model.summary()
#model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
#history = model.fit(X1_train,y1_train, validation_split=.1, class_weight=class_weights)
#pd.DataFrame(history.history).plot();
#p = model.predict(X1_test)
#scores(p,y1_test)
#model.evaluate(X1_test,y1_test)

# Need more time training this but will do it once we are done with all exams.

#Create a function to calculate f1 score as this is better than accuracy.
def scores(y_pred,y_true):
    rows,columns = y_pred.shape
    correct_dims = rows*columns

    TP = []
    TN = []
    FP = []
    FN = []

    for i in range(rows):
        for j in range(columns):
            if y_pred[i][j] > .50 and y_true[i][j]==1:
                TP.append(1)
            elif y_pred[i][j] < .50 and y_true[i][j]==1:
                FN.append(1)
            elif y_pred[i][j] < .50 and y_true[i][j]==0:
                TN.append(1)
            elif y_pred[i][j] > .50 and y_true[i][j]==0:
                FP.append(1)
            else:
                pass
            
    if sum(TP)+sum(FN)+sum(TN)+sum(FP) != correct_dims:
        print ('Wrong Sum.Check Again!')
    else:
        precision = sum(TP) / (sum(TP)+sum(FP))
        recall = sum(TP) / (sum(TP)+sum(FN))
        fone_score = 2*((precision*recall) / (precision+recall))  
    
    return ('F1 Score:',fone_score,'Precision:',precision,'Recall:',recall)


# Final F1 Score: 0.69