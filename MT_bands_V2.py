# %%
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error
import tensorflow as tf
import numpy as np
import utils as ut
import models256 as mo
import pandas as pd
import argparse


#python MT_bands.py -epo 1 -bs 500 -pat 5 -iter 1

# %%
parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument("-epo", dest = "epo", type = int)
parser.add_argument("-bs", dest = "bs", type = int)
parser.add_argument("-pat", dest = "pat", type = int)
parser.add_argument("-iter", dest = "iter", type = int)
args = parser.parse_args()


# %%

epo = args.epo
bs = args.bs
pat = args.pat
n_iter = args.iter

'''
epo = 1
bs = 500
pat = 5
n_iter = 1
'''

dff = pd.DataFrame()

#text    = ['High. vs River', 'Pasture vs Forest', 'Per. crop vs An. Crop', 'Pasture vs An. Crop', 'Pasture vs Per. Crop', 'Agri vs Grassland', 'Grassland vs Urban'] 
#dataset = [ ['Highway', 'River'], ['Pasture', 'Forest'], ['PermanentCrop', 'AnnualCrop'], ['Pasture', 'AnnualCrop'], ['Pasture', 'PermanentCrop'], ['agri', 'grassland'],  ['grassland', 'urban']]
text    = ['Agri vs Grassland', 'Grassland vs Urban']
dataset = [ ['agri', 'grassland'],  ['grassland', 'urban']]
text    = ['Agri vs Grassland'] 
dataset = [ ['agri', 'grassland']]  
datasets_dict = dict(zip(text, dataset))

# %%

for q in text:
    var = datasets_dict[q]
    err_up, err_b, mae_kt, err_kt = [[] for i in range(4)]
    err_mt, err_mt_tpd, err_mt_pfd = [[] for i in range(3)]
    err_pfd, err_tpd = [[] for i in range(2)]


    

    #EUROSAT

    #------------------------------------------------------------------------------------------------
    # Load train data with labels
    #v2
    input_size = (256, 256)
    train_rgb, train_nir, train_labels = ut.load_image_pairs_with_labels_s2_s1('v_2/split/train/', var,  input_size=input_size)

    test_rgb, test_nir, test_labels = ut.load_image_pairs_with_labels_s2_s1('v_2/split/test', var, input_size=input_size)
    #------------------------------------------------------------------------------------------------

    test_pri = np.concatenate([test_rgb, test_nir], axis=3)

    ran = np.random.randint(1000, size = n_iter)
    for k in ran:
        
        # Split train into train and validation
        train_rgb, val_rgb, train_nir, val_nir, train_labels, val_labels = train_test_split(
            train_rgb, train_nir, train_labels, test_size=0.2, random_state=k)

        
        #----------------------------------------------
        #UPPER
        #----------------------------------------------

        train_pri = np.concatenate([train_rgb, train_nir], axis=3)
        val_pri = np.concatenate([val_rgb, val_nir], axis=3)


        model = mo.fcnn256(input_shape = train_pri[0].shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_pri, train_labels,
                            validation_data=(val_pri, val_labels),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])

        pre = np.round(np.ravel(model.predict(test_pri)))
        err_up.append( 1 - accuracy_score(pre, test_labels))

        #Distillation training data
        pre_prob_upper = np.ravel(model.predict(train_pri))
        delta_i = np.array((train_labels == np.round(pre_prob_upper)))*1

        #Distillation val data
        pre_prob_upper_val = np.ravel(model.predict(val_pri))
        delta_i_val = np.array((val_labels == np.round(pre_prob_upper_val)))*1

        tf.keras.backend.clear_session()

        
        #----------------------------------------------
        #LOWER
        #----------------------------------------------

        model = mo.fcnn256(input_shape = train_rgb[0].shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_rgb, train_labels,
                            validation_data=(val_rgb, val_labels),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])


        pre = np.round(np.ravel(model.predict(test_rgb)))
        err_b.append( 1 - accuracy_score(pre, test_labels))

        tf.keras.backend.clear_session()


        #---------------------------------------------
        #KNOWLEDGE TRANSFER STANDARD
        #---------------------------------------------

        #AUTOENCODER
        model = mo.simple_unet256(train_rgb[0].shape)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae'])


        model.fit(train_rgb, train_nir,
                validation_data=(val_rgb, val_nir),
                epochs=epo,
                batch_size=bs,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])
        
        pre_nir = model.predict(test_rgb)
        mae_kt.append(mean_absolute_error(np.ravel(test_nir), np.ravel(pre_nir)))

        pre_train = model.predict(train_rgb)  
        pre_val = model.predict(val_rgb)    
        pre_test = model.predict(test_rgb)    

        train_concat = np.concatenate([train_rgb, pre_train], axis=3)
        val_concat = np.concatenate([val_rgb, pre_val], axis=3)
        test_concat = np.concatenate([test_rgb, pre_test], axis=3)
        

        

        #CLASSIFICATION
        model = mo.fcnn256(input_shape = train_concat[0].shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_concat, train_labels,
                            validation_data=(val_concat, val_labels),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])


        pre = np.round(np.ravel(model.predict(test_concat)))
        err_kt.append( 1 - accuracy_score(pre, test_labels))

        tf.keras.backend.clear_session()

        
        #---------------------------------------------
        #PFD
        #---------------------------------------------

        yy_PFD = np.column_stack([np.ravel(train_labels), np.ravel(pre_prob_upper)])
        yy_PFD_val= np.column_stack([np.ravel(val_labels), np.ravel(pre_prob_upper_val)])

        
        model = mo.fcnn256(input_shape = train_rgb[0].shape)            
        model.compile(loss= ut.loss_GD(1, 0.5), optimizer='adam', metrics=['accuracy'])
    
        #Fit the model
        model.fit(train_rgb, yy_PFD, 
                    validation_data=(val_rgb, yy_PFD_val),
                    epochs=epo, 
                    batch_size=bs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)], verbose = 0)      


        #Measure test error
        y_pre = np.ravel([np.round(i) for i in model.predict(test_rgb)])
        err_pfd.append(1-accuracy_score(test_labels, y_pre))

        tf.keras.backend.clear_session()



        #---------------------------------------------
        #TPD
        #---------------------------------------------
        yy_TPD = np.column_stack([np.ravel(train_labels), np.ravel(pre_prob_upper), delta_i])
        yy_TPD_val= np.column_stack([np.ravel(val_labels), np.ravel(pre_prob_upper_val), delta_i_val])

        
        model = mo.fcnn256(input_shape = train_rgb[0].shape)            
        model.compile(loss= ut.loss_TPD(1, 1, 0.5), optimizer='adam', metrics=['accuracy'])
    
        #Fit the model
        model.fit(train_rgb, yy_TPD, 
                    validation_data=(val_rgb, yy_TPD_val),
                    epochs=epo, 
                    batch_size=bs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)], verbose = 0)      


        #Measure test error
        y_pre = np.ravel([np.round(i) for i in model.predict(test_rgb)])
        err_tpd.append(1-accuracy_score(test_labels, y_pre))

        tf.keras.backend.clear_session()


        #---------------------------------------------
        #MT
        #---------------------------------------------
        mt_model = mo.MT_band256()
        mt_model.compile(optimizer='adam', loss=ut.loss_MT)

        train_label_ex = ut.expand_array(train_labels)    
        val_label_ex = ut.expand_array(val_labels)  

        y_MT = np.concatenate([train_nir, train_label_ex], axis = -1)
        y_MT_val = np.concatenate([val_nir, val_label_ex], axis = -1)

        # Entrenamiento
        mt_model.fit( train_rgb, y_MT, 
                    validation_data=(val_rgb, y_MT_val), 
                    epochs=epo, 
                    batch_size=bs,
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)], verbose = 0)

        pred = mt_model.predict(test_rgb)
        predictions = np.round(np.max(pred[:,:,:,1], axis = (1,2)))
        # Calculate errors
        err_mt.append(1 - accuracy_score(test_labels, predictions))

        tf.keras.backend.clear_session()


        #---------------------------------------------
        #MT PFD
        #---------------------------------------------
        mt_model = mo.MT_band256()
        mt_model.compile(optimizer='adam', loss= ut.loss_MT_PFD)

    
        pre_upper_ex = ut.expand_array(pre_prob_upper) 
        pre_upper_ex_val = ut.expand_array(pre_prob_upper_val) 


        y_MT_PFD = np.concatenate([train_nir, train_label_ex, pre_upper_ex], axis = -1)
        y_MT_PFD_val = np.concatenate([val_nir, val_label_ex, pre_upper_ex_val], axis = -1)

        # Entrenamiento
        mt_model.fit( train_rgb, y_MT_PFD, 
                    validation_data=(val_rgb, y_MT_PFD_val), 
                    epochs=epo, 
                    batch_size=bs,
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)], verbose = 0)

        pred = mt_model.predict(test_rgb)
        predictions = np.round(np.max(pred[:,:,:,1], axis = (1,2)))
        # Calculate errors
        err_mt_pfd.append(1 - accuracy_score(test_labels, predictions))
        
        tf.keras.backend.clear_session()


        #---------------------------------------------
        #MT TPD
        #---------------------------------------------
        mt_model = mo.MT_band256()
        mt_model.compile(optimizer='adam', loss=ut.loss_MT_TPD)

    
        delta_ex = ut.expand_array(delta_i) 
        delta_ex_val = ut.expand_array(delta_i_val) 

        y_MT = np.concatenate([train_nir, train_label_ex, pre_upper_ex, delta_ex], axis = -1)
        y_MT_val = np.concatenate([val_nir, val_label_ex, pre_upper_ex_val, delta_ex_val], axis = -1)

        # Entrenamiento
        mt_model.fit( train_rgb, y_MT, 
                    validation_data=(val_rgb, y_MT_val), 
                    epochs=epo, 
                    batch_size=bs,
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)], verbose = 0)

        pred = mt_model.predict(test_rgb)
        predictions = np.round(np.max(pred[:,:,:,1], axis = (1,2)))
        # Calculate errors
        err_mt_tpd.append(1 - accuracy_score(test_labels, predictions))
        
            
        tf.keras.backend.clear_session()
        
    #Save the results

    off = {'name': q,
            'err_up':np.round(np.mean(err_up), 4),
            'err_b':  np.round(np.mean(err_b), 4),
            'PFD': np.round(np.mean(err_pfd), 4),
            'TPD': np.round(np.mean(err_tpd), 4),
            'err_kt':  np.round(np.mean(err_kt), 4),
            'err_mt':  np.round(np.mean(err_mt), 4),
            'err_mt_pfd':  np.round(np.mean(err_mt_pfd), 4),
            'err_mt_tpd':  np.round(np.mean(err_mt_tpd), 4),
            'LUPI_pfd %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_pfd), 4)),
            'LUPI_tpd %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_tpd), 4)),
            'LUPI_KT %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_kt), 4)),
            'LUPI_MT %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_mt), 4)),
            'LUPI_MT_PFD%': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_mt_pfd), 4)),
            'LUPI_MT_TPD%': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_mt_tpd), 4)),
            'std_up':np.round(np.std(err_up), 4),
            'std_b':  np.round(np.std(err_b), 4),
            'std_PFD': np.round(np.std(err_pfd), 4),
            'std_TPD': np.round(np.std(err_tpd), 4),
            'std_kt':  np.round(np.std(err_kt), 4),
            'std_mt':  np.round(np.std(err_mt), 4),
            'std_mt_pfd':  np.round(np.std(err_mt_pfd), 4),
            'std_mt_tpd':  np.round(np.std(err_mt_tpd), 4)
            }   

    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)


dff.to_csv('Bands256_unet_' + str(epo) + '_' + str(bs) + '_' + str(pat)+ '_' + str(n_iter)+ '.csv')



# %%
