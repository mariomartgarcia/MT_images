# %%
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error
import tensorflow as tf
import numpy as np
import utils as ut
import models as mo
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, f1_score
import utils_multiclass as utm








#python MT_multiclass.py -PP SWIR -epo 1 -bs 500 -pat 5 -iter 1

# %%
parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument("-PP", dest = "pp", type = str)
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
PP = args.pp

'''
epo = 5
bs = 500
pat = 1
n_iter = 1
PP = 'NIR'
'''

dff = pd.DataFrame()

#text    = ['High. vs River', 'Pasture vs Forest', 'Per. crop vs An. Crop', 'Pasture vs An. Crop', 'Pasture vs Per. Crop'] 
#dataset = [ ['Highway', 'River'], ['Pasture', 'Forest'], ['PermanentCrop', 'AnnualCrop'], ['Pasture', 'AnnualCrop'], ['Pasture', 'PermanentCrop']]
#text    = ['High. vs River',  'Pasture vs An. Crop']
#dataset = [ ['Highway', 'River'], ['Pasture', 'AnnualCrop']]

#text    = ['High. vs River']
#dataset = [ ['Highway', 'River']]


text    = ['MultiClass']
dataset = [ ['Highway', 'River', 'Pasture', 'Forest', 'PermanentCrop', 'AnnualCrop']]

datasets_dict = dict(zip(text, dataset))


# %%

for q in text:
    var = datasets_dict[q]
    # Inicializar listas para error rate por clase y total para cada modelo
    err_up_classes = [[] for _ in range(len(var))]
    err_b_classes = [[] for _ in range(len(var))]
    err_kt_classes = [[] for _ in range(len(var))]
    err_kt_pfd_classes = [[] for _ in range(len(var))]
    err_kt_tpd_classes = [[] for _ in range(len(var))]
    err_mt_classes = [[] for _ in range(len(var))]
    err_mt_pfd_classes = [[] for _ in range(len(var))]
    err_mt_tpd_classes = [[] for _ in range(len(var))]
    err_pfd_classes = [[] for _ in range(len(var))]
    err_tpd_classes = [[] for _ in range(len(var))]

    err_up_total = []
    err_b_total = []
    err_kt_total = []
    err_kt_pfd_total = []
    err_kt_tpd_total = []
    err_mt_total = []
    err_mt_pfd_total = []
    err_mt_tpd_total = []
    err_pfd_total = []
    err_tpd_total = []


    #EUROSAT
    print(q)
    input_size = (64, 64)
    train_rgbRAW, train_nirRAW, train_labelsRAW = ut.load_image_pairs_with_labels('eurosat/split/train/', var, input_size=input_size, priv = PP)

    # Load test data with labels
    test_rgb, test_nir, test_labels = ut.load_image_pairs_with_labels('eurosat/split/test/', var, input_size=input_size, priv = PP)
    test_pri = np.concatenate([test_rgb, test_nir], axis=3)


    ran = np.random.randint(1000, size = n_iter)
    for k in ran:

        # Split train into train and validation
        train_rgb, val_rgb, train_nir, val_nir, train_labels, val_labels = train_test_split(
            train_rgbRAW, train_nirRAW, train_labelsRAW, test_size=0.2, random_state=k)

        
        #----------------------------------------------
        #UPPER
        #----------------------------------------------

        train_pri = np.concatenate([train_rgb, train_nir], axis=3)
        val_pri = np.concatenate([val_rgb, val_nir], axis=3)


        model = mo.fcnn_multi(n = len(var), input_shape = train_pri[0].shape)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



        # Train the model for the current fold with EarlyStopping
        model.fit(train_pri, train_labels,
                            validation_data=(val_pri, val_labels),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])
        pre = np.ravel(np.argmax(model.predict(test_pri), axis=1))
        # Calcular error rate por clase para el modelo UPPER



        err_up_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_up_classes)
        err_up_total.append(1 - accuracy_score(test_labels, pre))



        

        #Distillation training data
        pre_prob_upper = np.ravel(np.argmax(model.predict(train_pri), axis=1))
        delta_i = np.array((train_labels == pre_prob_upper))*1

        #Distillation val data
        pre_prob_upper_val = np.ravel(np.argmax(model.predict(val_pri), axis=1))
        delta_i_val = np.array((val_labels == pre_prob_upper_val))*1 

        
        #----------------------------------------------
        #LOWER
        #----------------------------------------------

        model =  mo.fcnn_multi(n = len(var), input_shape = train_rgb[0].shape)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_rgb, train_labels,
                            validation_data=(val_rgb, val_labels),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])

        pre = np.ravel(np.argmax(model.predict(test_rgb), axis=1))
        # Calcular error rate por clase para el modelo LOWER
        err_b_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_b_classes)
        err_b_total.append(1 - accuracy_score(test_labels, pre))


        #---------------------------------------------
        #KNOWLEDGE TRANSFER STANDARD
        #---------------------------------------------


        #AUTOENCODER
        model = mo.simple_unet(train_rgb[0].shape)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae'])


        model.fit(train_rgb, train_nir,
                validation_data=(val_rgb, val_nir),
                epochs=epo,
                batch_size=bs,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])
        
        pre_nir = model.predict(test_rgb)
        #mae_kt.append(mean_absolute_error(np.ravel(test_nir), np.ravel(pre_nir)))

        pre_train = model.predict(train_rgb)  
        pre_val = model.predict(val_rgb)    
        pre_test = model.predict(test_rgb)    

        train_concat = np.concatenate([train_rgb, pre_train], axis=3)
        val_concat = np.concatenate([val_rgb, pre_val], axis=3)
        test_concat = np.concatenate([test_rgb, pre_test], axis=3)
        
        

        #CLASSIFICATION
        model = mo.fcnn_multi(n = len(var), input_shape = train_concat[0].shape)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_concat, train_labels,
                            validation_data=(val_concat, val_labels),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])

        pre = np.ravel(np.argmax(model.predict(test_concat), axis=1))
        # Calcular error rate por clase para el modelo KT
        err_kt_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_kt_classes)
        err_kt_total.append(1 - accuracy_score(test_labels, pre))


        #---------------------------------------------
        #KNOWLEDGE TRANSFER PFD
        #---------------------------------------------
        yy_PFD = np.column_stack([np.ravel(train_labels), np.ravel(pre_prob_upper)])
        yy_PFD_val= np.column_stack([np.ravel(val_labels), np.ravel(pre_prob_upper_val)])

        #AUTOENCODER
        model = mo.simple_unet(train_rgb[0].shape)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae'])


        model.fit(train_rgb, train_nir,
                validation_data=(val_rgb, val_nir),
                epochs=epo,
                batch_size=bs,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])
        
        pre_nir = model.predict(test_rgb)
        #mae_kt_pfd.append(mean_absolute_error(np.ravel(test_nir), np.ravel(pre_nir)))

        pre_train = model.predict(train_rgb)  
        pre_val = model.predict(val_rgb)    
        pre_test = model.predict(test_rgb)    

        train_concat = np.concatenate([train_rgb, pre_train], axis=3)
        val_concat = np.concatenate([val_rgb, pre_val], axis=3)
        test_concat = np.concatenate([test_rgb, pre_test], axis=3)
        
        

        #CLASSIFICATION
        model =  mo.fcnn_multi(n = len(var), input_shape = train_concat[0].shape)
        model.compile(optimizer='adam', loss= utm.loss_GD_mc(0.5, len(var)), metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_concat, yy_PFD,
                            validation_data=(val_concat, yy_PFD_val),
                            epochs=epo,
                            batch_size=bs,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])

        pre = np.ravel(np.argmax(model.predict(test_concat), axis=1))
        # Calcular error rate por clase para el modelo KT PFD
        err_kt_pfd_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_kt_pfd_classes)
        err_kt_pfd_total.append(1 - accuracy_score(test_labels, pre))




        #---------------------------------------------
        #KNOWLEDGE TRANSFER TPD
        #---------------------------------------------
        yy_TPD = np.column_stack([np.ravel(train_labels), np.ravel(pre_prob_upper), delta_i])
        yy_TPD_val= np.column_stack([np.ravel(val_labels), np.ravel(pre_prob_upper_val), delta_i_val])


        #AUTOENCODER
        model = mo.simple_unet(train_rgb[0].shape)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['mae'])


        model.fit(train_rgb, train_nir,
                validation_data=(val_rgb, val_nir),
                epochs=epo,
                batch_size=bs,
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])
        
        pre_nir = model.predict(test_rgb)
        #mae_kt_tpd.append(mean_absolute_error(np.ravel(test_nir), np.ravel(pre_nir)))

        pre_train = model.predict(train_rgb)  
        pre_val = model.predict(val_rgb)    
        pre_test = model.predict(test_rgb)    

        train_concat = np.concatenate([train_rgb, pre_train], axis=3)
        val_concat = np.concatenate([val_rgb, pre_val], axis=3)
        test_concat = np.concatenate([test_rgb, pre_test], axis=3)
        
        

        #CLASSIFICATION
        model = mo.fcnn_multi(n = len(var), input_shape = train_concat[0].shape)
        model.compile(optimizer='adam', loss=utm.loss_TPD_mc(1, 0.5, len(var)), metrics=['accuracy'])


        # Train the model for the current fold with EarlyStopping
        model.fit(train_concat, yy_TPD,
                            validation_data=(val_concat, yy_TPD_val),
                            epochs=epo,
                            batch_size=2,
                            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)])

        pre = np.ravel(np.argmax(model.predict(test_concat), axis=1))
        # Calcular error rate por clase para el modelo KT TPD
        err_kt_tpd_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_kt_tpd_classes)
        err_kt_tpd_total.append(1 - accuracy_score(test_labels, pre))

        
        #---------------------------------------------
        #PFD
        #---------------------------------------------
        
        model = mo.fcnn_multi(n = len(var), input_shape = train_rgb[0].shape)        
        model.compile(loss= utm.loss_GD_mc(0.5, len(var)), optimizer='adam', metrics=['accuracy'])
    
        #Fit the model
        model.fit(train_rgb, yy_PFD, 
                    validation_data=(val_rgb, yy_PFD_val),
                    epochs=epo, 
                    batch_size=bs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)], verbose = 0)      

        pre = np.ravel(np.argmax(model.predict(test_rgb), axis=1))
        # Calcular error rate por clase para el modelo PFD  
        err_pfd_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_pfd_classes)
        err_pfd_total.append(1 - accuracy_score(test_labels, pre))
  


        #---------------------------------------------
        #TPD
        #---------------------------------------------

        model = mo.fcnn_multi(n = len(var), input_shape = train_rgb[0].shape)                 
        model.compile(loss= utm.loss_TPD_mc(1, 0.5, len(var)), optimizer='adam', metrics=['accuracy'])
    
        #Fit the model
        model.fit(train_rgb, yy_TPD, 
                    validation_data=(val_rgb, yy_TPD_val),
                    epochs=epo, 
                    batch_size=bs,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)], verbose = 0)      

        pre = np.ravel(np.argmax(model.predict(test_rgb), axis=1))
        # Calcular error rate por clase para el modelo TPD
        err_tpd_classes = utm.compute_class_error_rates(test_labels, pre, len(var), err_tpd_classes)
        err_tpd_total.append(1 - accuracy_score(test_labels, pre))
  


        #---------------------------------------------
        #MT
        #---------------------------------------------

        mt_model = mo.MT_band_multi(len(var))
        mt_model.compile(optimizer='adam', loss=utm.loss_MT_mc(len(var)))

        train_label_ex = ut.expand_array(train_labels)    
        val_label_ex = ut.expand_array(val_labels)  

        y_MT = np.concatenate([train_nir, train_label_ex], axis = -1)
        y_MT_val = np.concatenate([val_nir, val_label_ex], axis = -1)

        # Entrenamiento
        mt_model.fit( train_rgb, y_MT, 
                    validation_data=(val_rgb, y_MT_val), 
                    epochs=epo, 
                    batch_size=2,
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',  patience=pat)], verbose = 0)

        pred = mt_model.predict(test_rgb)
        # Extrae los mapas de probabilidad de clase (canales 1 a n)
        class_maps = pred[:, :, :, 1:len(var)+1]  # shape: (num_samples, 64, 64, n)


        # Promedia espacialmente cada clase
        class_scores = np.max(class_maps, axis=(1, 2))  # shape: (num_samples, n)
        # Predice la clase con mayor score promedio
        predictions = np.argmax(class_scores, axis=1)  # shape: (num_samples,)

        # Calcular error rate por clase para el modelo MT
        err_mt_classes = utm.compute_class_error_rates(test_labels, predictions, len(var), err_mt_classes)
        err_mt_total.append(1 - accuracy_score(test_labels, predictions))

        #---------------------------------------------
        #MT PFD
        #---------------------------------------------


        mt_model = mo.MT_band_multi(len(var))
        mt_model.compile(optimizer='adam', loss= utm.loss_MT_PFD_mc(len(var)))

    
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
        # Extrae los mapas de probabilidad de clase (canales 1 a n)
        class_maps = pred[:, :, :, 1:len(var)+1]  # shape: (num_samples, 64, 64, n)

        # Promedia espacialmente cada clase
        class_scores = np.max(class_maps, axis=(1, 2))  # shape: (num_samples, n)
        # Predice la clase con mayor score promedio
        predictions = np.argmax(class_scores, axis=1)  # shape: (num_samples,)

        # Calcular error rate por clase para el modelo MT PFD
        err_mt_pfd_classes = utm.compute_class_error_rates(test_labels, predictions, len(var), err_mt_pfd_classes)
        err_mt_pfd_total.append(1 - accuracy_score(test_labels, predictions))   



        #---------------------------------------------
        #MT TPD
        #---------------------------------------------

        mt_model = mo.MT_band_multi(len(var))
        mt_model.compile(optimizer='adam', loss= utm.loss_MT_TPD_mc(0.5, len(var)))

    
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
        # Extrae los mapas de probabilidad de clase (canales 1 a n)
        class_maps = pred[:, :, :, 1:len(var)+1]  # shape: (num_samples, 64, 64, n)

        # Promedia espacialmente cada clase
        class_scores = np.max(class_maps, axis=(1, 2))  # shape: (num_samples, n)
        # Predice la clase con mayor score promedio
        predictions = np.argmax(class_scores, axis=1)  # shape: (num_samples,)   

        # Calcular error rate por clase para el modelo MT TPD
        err_mt_tpd_classes = utm.compute_class_error_rates(test_labels, predictions, len(var), err_mt_tpd_classes)
        err_mt_tpd_total.append(1 - accuracy_score(test_labels, predictions))

            
        tf.keras.backend.clear_session()
        
    #Save the results

    # Calcular medias de error rate por clase para cada modelo y añadirlas al diccionario off
    def mean_class_errors(err_classes, var, prefix):
        # err_classes: lista de listas, cada sublista es para una clase
        # var: lista de nombres de clases
        # prefix: prefijo para la métrica, ej: 'err_mt_tpd'
        out = {}
        for idx, class_name in enumerate(var):
            # Calcula la media de error para la clase idx
            mean_err = np.round(np.mean(err_classes[idx]), 3)
            key = f"{prefix}_{class_name.lower()}"
            out[key] = mean_err
        return out
    
    # Añadir desviación estándar de error rate por clase para cada modelo
    def std_class_errors(err_classes, var, prefix):
        out = {}
        for idx, class_name in enumerate(var):
            std_err = np.round(np.std(err_classes[idx]), 3)
            key = f"std_{prefix}_{class_name.lower()}"
            out[key] = std_err
        return out

    off = {
        'name': q,
        'err_up': np.round(np.mean(err_up_total), 3),
        'err_b': np.round(np.mean(err_b_total), 3),
        'PFD': np.round(np.mean(err_pfd_total), 3),
        'TPD': np.round(np.mean(err_tpd_total), 3),
        'kt': np.round(np.mean(err_kt_total), 3),
        'kt_pfd': np.round(np.mean(err_kt_pfd_total), 3),
        'kt_tpd': np.round(np.mean(err_kt_tpd_total), 3),
        'mt': np.round(np.mean(err_mt_total), 3),
        'mt_pfd': np.round(np.mean(err_mt_pfd_total), 3),
        'mt_tpd': np.round(np.mean(err_mt_tpd_total), 3),
        'std_up': np.round(np.std(err_up_total), 3),
        'std_b': np.round(np.std(err_b_total), 3),
        'std_PFD': np.round(np.std(err_pfd_total), 3),
        'std_TPD': np.round(np.std(err_tpd_total), 3),
        'std_kt': np.round(np.std(err_kt_total), 3),
        'std_kt_pfd': np.round(np.std(err_kt_pfd_total), 3),
        'std_kt_tpd': np.round(np.std(err_kt_tpd_total), 3),
        'std_mt': np.round(np.std(err_mt_total), 3),
        'std_mt_pfd': np.round(np.std(err_mt_pfd_total), 3),
        'std_mt_tpd': np.round(np.std(err_mt_tpd_total), 3),
        # Puedes añadir aquí más métricas globales si las necesitas
    }

    # Añadir error rate por clase para cada modelo
    off.update(mean_class_errors(err_up_classes, var, 'err_up'))
    off.update(mean_class_errors(err_b_classes, var, 'err_b'))
    off.update(mean_class_errors(err_kt_classes, var, 'err_kt'))
    off.update(mean_class_errors(err_kt_pfd_classes, var, 'err_kt_pfd'))
    off.update(mean_class_errors(err_kt_tpd_classes, var, 'err_kt_tpd'))
    off.update(mean_class_errors(err_mt_classes, var, 'err_mt'))
    off.update(mean_class_errors(err_mt_pfd_classes, var, 'err_mt_pfd'))
    off.update(mean_class_errors(err_mt_tpd_classes, var, 'err_mt_tpd'))
    off.update(mean_class_errors(err_pfd_classes, var, 'err_pfd'))
    off.update(mean_class_errors(err_tpd_classes, var, 'err_tpd'))



    off.update(std_class_errors(err_up_classes, var, 'err_up'))
    off.update(std_class_errors(err_b_classes, var, 'err_b'))
    off.update(std_class_errors(err_kt_classes, var, 'err_kt'))
    off.update(std_class_errors(err_kt_pfd_classes, var, 'err_kt_pfd'))
    off.update(std_class_errors(err_kt_tpd_classes, var, 'err_kt_tpd'))
    off.update(std_class_errors(err_mt_classes, var, 'err_mt'))
    off.update(std_class_errors(err_mt_pfd_classes, var, 'err_mt_pfd'))
    off.update(std_class_errors(err_mt_tpd_classes, var, 'err_mt_tpd'))
    off.update(std_class_errors(err_pfd_classes, var, 'err_pfd'))
    off.update(std_class_errors(err_tpd_classes, var, 'err_tpd'))


    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)


dff.to_csv('Multi_EuBands_unet_' + PP + '_' + str(epo) + '_' + str(bs) + '_' + str(pat)+ '_' + str(n_iter)+ '.csv')


# %%
