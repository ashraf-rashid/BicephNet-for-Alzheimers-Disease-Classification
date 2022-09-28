import pickle
import pandas as pd
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import sys
import random
import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Bidirectional, Lambda, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers

class Slices:
    def __init__(self, class_ID, file_name, class_label, is_selected_for_batch = False):
        self.class_ID = class_ID
        self.class_label = class_label
        self.file_name = file_name
        self.is_selected_for_batch = is_selected_for_batch
        
    def is_selected(self):
        return self.is_selected_for_batch
        #return True
    
    def get_class_ID(self):
        return self.class_ID
    
    def select_slice(self):
        
        if self.is_selected_for_batch == False:        
            self.is_selected_for_batch = True
            return True
        else:
            return False
    
    def deselect_slice(self):
        self.is_selected_for_batch = False
    
class SampledSlices:
    def __init__(self, class_ID, class_label, slice_list):
        self.slice_list = []
        
        for i in slice_list:
            tmp_slice = Slices(class_ID, i, class_label)
            self.slice_list.append(tmp_slice)
         
        self.unselected_slice_index = list(range(0,len(self.slice_list)))
        self.is_all_slices_selected = False
         
    def select_slices(self, indexes): #index: index of slice_list to be selected
        selected_slices = []
        
        for i in indexes:
            #print(i)
            val = self.slice_list[i].select_slice()
            
            if val == False:
                continue
            else:           
                selected_slices.append(self.slice_list[i])
                self.unselected_slice_index.remove(i) #remove index of slice from unselected list

        
        if len(self.unselected_slice_index) <= 0:
            self.is_all_slices_selected = True
            
        return selected_slices
         
    def sample_unselected_slices(self,K): #K - number of slices to sample without replacement
        
        sampled_indexes = None
        
        try:
            sampled_indexes = random.sample(self.unselected_slice_index, K)
        except:
            
            if len(self.unselected_slice_index) > 0:
                sampled_indexes = random.sample(self.unselected_slice_index, len(self.unselected_slice_index))
            else: 
                self.is_all_slices_selected = True
                return -1
        
        selected_slices = self.select_slices(sampled_indexes)
        
        return selected_slices
    
    def reset(self):
        for s in self.slice_list:
            s.deselect_slice()
        
        self.unselected_slice_index = list(range(0,len(self.slice_list)))
        self.is_all_slices_selected = False

        
class Subjects:
    
    def __init__(self, class_ID, class_label, slice_list):
        self.class_ID = class_ID
        self.class_label = class_label
        self.is_selected_for_batch = False #to show whether the subject has been completely selected in an epoch
        self.slices = SampledSlices(class_ID, class_label, slice_list)
        
    def is_selected(self):
        return self.is_selected_for_batch
    
    def select_subject(self):
        if self.slices.is_all_slices_selected:
            self.is_selected_for_batch = True  
            return True
        else:
            return False      

    def deselect_subject(self):
        self.is_selected_for_batch = False
    
class SampledSubjects:
    ##subject_list: list of lists containing all slices of each subject
    def __init__(self, class_ID, class_label, subject_list): 
        self.subject_list = []
        
        for i,j,k in zip(class_ID, class_label, subject_list):
            s_subject = Subjects(i, j, k)
            self.subject_list.append(s_subject)
            
        self.unselected_subject_index = list(range(0, len(self.subject_list)))
        
    def select_subjects(self, indexes, K):
        selected_slices = []
        
        for i in indexes:
            cur_sub = self.subject_list[i]
            sampled_slices = cur_sub.slices.sample_unselected_slices(K)
            
            if sampled_slices == -1: 
                v = cur_sub.select_subject()
                
                if v == False:
                    print('unable to select subject ', str(self.class_ID))
                    sys.exit(1)
                else:
                    self.unselected_subject_index.remove(i)
            else:
                for s in sampled_slices:
                    selected_slices.append(s)

                if cur_sub.slices.is_all_slices_selected:
                    v = cur_sub.select_subject()

                    if v == False:
                        print('unable to select subject ', str(self.class_ID))
                        sys.exit(1)
                    else:
                        self.unselected_subject_index.remove(i)

        return selected_slices
    
    def sample_unselected_subjects(self, P, K):
        sampled_indexes = None
        idx_left = len(self.unselected_subject_index)
        
        if idx_left >= P:
            sampled_indexes = random.sample(self.unselected_subject_index, P)
        elif idx_left < P:
            if idx_left > 0:
                sampled_indexes = random.sample(self.unselected_subject_index, idx_left)
            else:
                self.reset_subjects()
                sampled_indexes = random.sample(self.unselected_subject_index, P)
                
        selected_slices = self.select_subjects(sampled_indexes, K)
        return selected_slices
        
        
    def reset_subjects(self):
        for s in self.subject_list:            
            s.slices.reset()
            s.deselect_subject()
            
        self.unselected_subject_index = list(range(0, len(self.subject_list)))

class DataGenerator_CV(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,  sampled_subjects, slice_per_subject, datapath, P = 25, K = 4, dim=(256,256), n_channels=3):
        'Initialization'
        self.dim = dim
        self.batch_size = P * K
        self.sampled_subjects = sampled_subjects
        self.n_channels = n_channels
        self.on_epoch_end
        self.P = P
        self.K = K
        self.datapath = datapath
        self.slice_per_subject = slice_per_subject
        self.num_iters = np.ceil((self.slice_per_subject * len(self.sampled_subjects.subject_list)) / self.batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print(self.sampled_subjects.subject_list)
        l = self.num_iters
        return int(l)
        

    def __getitem__(self, index):
        #print('index: ', index)
        sampled_slices = self.sampled_subjects.sample_unselected_subjects(self.P, self.K)
        X, y = self.__data_generation(sampled_slices)
        #print(X.shape, y.shape)
        return X, y

    def on_epoch_end(self):
        #reset all the subjecs
        self.sampled_subjects.reset_subjects()
        self.indexes = np.arange(self.num_iters)

    def __data_generation(self, list_IDs_temp):
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        cur_batch_len = len(list_IDs_temp)
        X = np.empty((cur_batch_len, *self.dim, self.n_channels))
        y = np.empty((cur_batch_len,))
        y1 = np.empty((cur_batch_len,))
        
        for i, s_slice in enumerate(list_IDs_temp):
            fname = s_slice.file_name
            class_ID = s_slice.class_ID
            class_label = s_slice.class_label
            final_label = None

            if class_label == 'MCI':
                final_label = 0
            else:
                final_label = 1


            tmp = np.load(self.datapath + "/" + fname)
            
            X[i,] = tmp
            y[i,] = class_ID
            y1[i,] = final_label

            #print(final_label)

            #import sys
            #sys.exit(1)
        # Generate data
        #print(X.shape, y.shape)
        return X, [y,y1]

    
#returns a list  of  hyperparamter settings
def load_hyperparameter_settings(sampler_file):
    hyp_list = []
    
    with open(sampler_file, "rb") as obj:
        for i in range(10):
            hyp_list.append(pickle.load(obj))
            
    return hyp_list

#returns resNet base_model
def load_base_model_vgg16(img_width,img_height,num_channels,weight_init='imagenet',include_fc_layers=False):
    base_model = applications.VGG16(include_top=include_fc_layers,weights=weight_init, input_shape=(img_width, img_height, num_channels), input_tensor=None, pooling=None)
                                    
    return base_model

# set the first num_layers to nontrainable
# model - an instance of Keras Model
# => model is the final model (base_model added with fully connected layers)

def set_nontrainable_layers(num_layers, model):
    for layer in model.layers[:num_layers]:
        layer.trainable = False
        
    return model

#returns the dict of cross validation settings

def load_cross_validation_settings(cv_file):
    cv_setting = None
    with open(cv_file, "rb") as obj:
        cv_setting = pickle.load(obj)
        
    return cv_setting
    

def save_model_history(history, history_path):    
    import pickle
    
    with open(history_path+"/Bicephalus_MCI_vs_AD_1365_margin_pat2_1_29_10_21_axial.pkl", 'wb') as handle:
        pickle.dump(history.history, handle)
    return     
#     df_train_loss = pd.DataFrame(history.history['loss'])
#     df_train_loss.columns = ['train_loss']
#     df_val_loss = pd.DataFrame(history.history['val_loss'])
#     df_val_loss.columns = ['validation_loss']
#     df_history = pd.concat([df_train_loss,df_val_loss], axis=1)
#     df_history.to_csv(history_path+"/AD_1365_margin_1_14_04_21_coronal.csv", index=False)
        
    
def fit_generator(model, training_generator, validation_generator, checkpoint_path):    
    cb_save_path = ''
    lrate_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,verbose=1,min_lr=0.0000001)
    mc = tf.keras.callbacks.ModelCheckpoint('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/CN_vs_AD/model_checkpoints/Bicephalus_MCI_vs_AD_1365_margin_pat2_1_29_10_21_axial_{epoch:08d}.h5', 
                                     save_weights_only=True, period=1)
    
    history = model.fit(
                training_generator,
                epochs = 65,
                verbose = 2,
                validation_data = validation_generator,
                callbacks = [lrate_reduce,mc]
            )
    return history 

def fit_cross_validation_triplet(model, cv_file, generator_params_dict, datapath, checkpoint_path, history_path):
    cv_setting = load_cross_validation_settings(cv_file)   
    full_train_dict = cv_setting['train']
    full_val_dict = cv_setting['validation']
    
    train_subject_id = full_train_dict['subject_dict']
    train_subject_group = full_train_dict['subject_group']
    train_subject_slices = full_train_dict['subject_slices']
    train_class_ID = []
    train_class_label = []
    train_subj_slices = []
        
    for idx in train_subject_id.keys():
        tmp_id = train_subject_id[idx]
        tmp_grp = train_subject_group[idx]
        tmp_slices = train_subject_slices[idx]
        
        train_class_ID.append(tmp_id)
        train_class_label.append(tmp_grp)
        train_subj_slices.append(tmp_slices)
    
    slice_per_subject = len(train_subject_slices[idx])
    
    #create training generator
    train_sampled_subjects = SampledSubjects(train_class_ID, train_class_label, train_subj_slices)   
    
    val_subject_id = full_val_dict['subject_dict']
    val_subject_group = full_val_dict['subject_group']
    val_subject_slices = full_val_dict['subject_slices']
    val_class_ID = []
    val_class_label = []
    val_subj_slices = []
    
    for idx in val_subject_id.keys():
        tmp_id = val_subject_id[idx]
        tmp_grp = val_subject_group[idx]
        tmp_slices = val_subject_slices[idx]
        
        val_class_ID.append(tmp_id)
        val_class_label.append(tmp_grp)
        val_subj_slices.append(tmp_slices)
        
    #create validation generator
    val_sampled_subjects = SampledSubjects(val_class_ID, val_class_label, val_subj_slices)   
    
    training_generator = DataGenerator_CV(train_sampled_subjects, slice_per_subject, datapath, **generator_params_dict)
    validation_generator = DataGenerator_CV(val_sampled_subjects, slice_per_subject, datapath, **generator_params_dict)
    
#     for x,y in training_generator:
#         print(x.shape, y.shape)
    
    history = fit_generator(model, training_generator, validation_generator, checkpoint_path)
    save_model_history(history,history_path)
    return    





img_width = 121
img_height = 145
channels = 3

datapath = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1'
params_dict = {'dim':(img_width,img_height),
                   'n_channels': 3,
                   'P': 20,
                   'K': 4,
                   #'shuffle':True
                  }
history_path = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/history'
checkpoint_path = '/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/model_checkpoints'
cv_file = '/media/iitindmaths/Seagate_Expansion_Drive/Alz_MCI_vs_AD_1365_each_full_subject_list.pkl'

    
#clf - bicephalus
vgg = tf.keras.applications.VGG16(include_top=False,input_shape=(img_width, img_height, channels))
vgg_l1 = Conv2D(256, (1,1), padding='same')(vgg.output)
vgg_l2 = Conv2D(128, (1,1), padding='same')(vgg_l1)
vgg_l3 = Conv2D(64, (1,1), padding='same')(vgg_l2)
vgg_flat = Flatten()(vgg_l3)
vgg_dense = Dense(64, activation=None)(vgg_flat)
vgg_lambda = Lambda(lambda x: tf.math.l2_normalize(x, axis=1),name='triplet_nw')(vgg_dense)


fc1 = Dense(64, activation='relu')(vgg_flat)
vgg_cre = Dense(1, activation='sigmoid', name='binary_cre')(fc1)

vgg_concat = tf.keras.layers.Concatenate()([vgg_lambda, vgg_cre])


final_dense1 = Dense(32, activation='relu')(vgg_concat)
final_dense2 = Dense(1, activation='sigmoid',name='binary_final')(final_dense1)

losses = {'triplet_nw':tfa.losses.TripletSemiHardLoss(),
         'binary_final':tf.keras.losses.BinaryCrossentropy(from_logits=False)}

lossWeights = {"triplet_nw": 1, "binary_final":1.0}
metrics = {"triplet_nw":tfa.losses.TripletSemiHardLoss() , "binary_final":"accuracy"}

model = Model(inputs = vgg.input, outputs = [vgg_lambda,final_dense2])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=losses,loss_weights=lossWeights,metrics=metrics)

#sys.exit(1)
#fit_cross_validation_triplet(model, cv_file, params_dict, datapath, checkpoint_path, history_path)

#testing data inference
params_dict = {'dim':(img_width,img_height),
                   'n_channels': 3,
                   'batch_size': 1,
                   'n_classes': 2,
                   'shuffle':True
                  }

model.load_weights('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/CN_vs_AD/model_checkpoints/Bicephalus_MCI_vs_AD_1365_margin_pat2_1_29_10_21_axial_00000024.h5')

cv_setting = load_cross_validation_settings(cv_file)
full_test_dict = cv_setting['test']
test_subject_id = full_test_dict['subject_dict']
test_subject_group = full_test_dict['subject_group']
test_subject_slices = full_test_dict['subject_slices']
test_subject_fnames = list(test_subject_id.keys())
slice_per_subject = len(test_subject_slices[test_subject_fnames[0]])

pred_list = []
c = 1
for k in test_subject_slices.keys():
    v = test_subject_slices[k]
    pred_array = []
    for i in v:
        ip = np.load('/media/iitindmaths/Seagate_Expansion_Drive/Bup_Backup/SPM/alzheimers-disease/MCI_vs_AD/npy_large/smwp1/' + i)
        #print(ip.shape)
        ip = ip.reshape((1,121,145,3))
        pred = model.predict(ip)
        #print(pred)
        if pred[1] > 0.5:
            val = 1
        else:
            val = 0
        
        pred_array.append(val)
    
    pred_list.append(pred_array)
    print(c)    
    c = c + 1


final_pred = []
for pred in pred_list:
	count_1 = pred.count(1)
	count_0 = pred.count(0)

	if count_1 > count_0:
		final_pred.append(1)		
	else:
		final_pred.append(0) #more specific
        
label_list = []
for k in test_subject_slices.keys():
	v = test_subject_group[k]

	if v == 'AD':
		label = 1
	else:
		label = 0
	
	label_list.append(label)

Y_true = np.array(label_list)
Y_pred = np.array(final_pred)

print(accuracy_score(Y_true, Y_pred))
        
print(vgg_lambda.shape)
print('reached')    
