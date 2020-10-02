#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ouranos
"""

import tensorflow as tf
import tensorflow.keras as keras

#from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model
import gc


def create_model17(num_dense_features, lr=0.002):
    tf.keras.backend.clear_session()
    gc.collect()

    # Dense input
    dense_input = Input(shape=(num_dense_features, ), name='dense1')

    # Embedding input
    wday_input = Input(shape=(1,), name='wday')
#     month_input = Input(shape=(1,), name='month')
#     year_input = Input(shape=(1,), name='year')
    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
#    item_id_input = Input(shape=(1,), name='item_id')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')


    wday_emb = Flatten()(Embedding(7, 2)(wday_input))
#     month_emb = Flatten()(Embedding(12, 2)(month_input))
#     year_emb = Flatten()(Embedding(6, 2)(year_input))
    event_name_1_emb = Flatten()(Embedding(31, 2)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, 2)(event_type_1_input))
    event_name_2_emb = Flatten()(Embedding(5, 2)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, 2)(event_type_2_input))

#    item_id_emb = Flatten()(Embedding(3049, 6)(item_id_input))
    dept_id_emb = Flatten()(Embedding(7, 2)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 3)(store_id_input))
    cat_id_emb = Flatten()(Embedding(3, 2)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 2)(state_id_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    x = concatenate([dense_input, 
                     wday_emb, #month_emb, year_emb, 
                     event_name_1_emb, event_type_1_emb, 
                     event_name_2_emb, event_type_2_emb, 
#                      item_id_emb, 
                     dept_id_emb, store_id_emb,
                     cat_id_emb, state_id_emb])
    

    x = Dense(256*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256*2, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(4*2, activation="relu")(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(1, activation="linear", name='output')(x)

    inputs = {"dense1": dense_input,
              "wday": wday_input, 
#               "month": month_input, "year": year_input, 
              "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,
              "event_name_2": event_name_2_input, "event_type_2": event_type_2_input,
#               "item_id": item_id_input, 
              "dept_id": dept_id_input, "store_id": store_id_input, 
              "cat_id": cat_id_input, "state_id": state_id_input}

    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(loss=keras.losses.mean_squared_error,
                  metrics=["mse"],
                  optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


def create_model17noEN1EN2(num_dense_features, lr=0.002):
    tf.keras.backend.clear_session()
    gc.collect()

    # Dense input
    dense_input = Input(shape=(num_dense_features, ), name='dense1')

    # Embedding input
    wday_input = Input(shape=(1,), name='wday')
#    month_input = Input(shape=(1,), name='month')
#    year_input = Input(shape=(1,), name='year')
#     event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
#     event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
#    item_id_input = Input(shape=(1,), name='item_id')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')

    wday_emb = Flatten()(Embedding(7, 2)(wday_input))
#     month_emb = Flatten()(Embedding(12, 2)(month_input))
#     year_emb = Flatten()(Embedding(6, 2)(year_input))
#     event_name_1_emb = Flatten()(Embedding(31, 2)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, 2)(event_type_1_input))
#     event_name_2_emb = Flatten()(Embedding(5, 2)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, 2)(event_type_2_input))

#    item_id_emb = Flatten()(Embedding(3049, 6)(item_id_input))
    dept_id_emb = Flatten()(Embedding(7, 2)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 3)(store_id_input))
    cat_id_emb = Flatten()(Embedding(3, 2)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 2)(state_id_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    x = concatenate([dense_input, 
                     wday_emb, #month_emb, year_emb, 
#                      event_name_1_emb, 
                     event_type_1_emb, 
#                      event_name_2_emb, 
                     event_type_2_emb, 
#                      item_id_emb, 
                     dept_id_emb, store_id_emb,
                     cat_id_emb, state_id_emb])
    
    x = Dense(256*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256*2, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(4*2, activation="relu")(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(1, activation="linear", name='output')(x)

    inputs = {"dense1": dense_input,
              "wday": wday_input, 
#               "month": month_input, "year": year_input, 
#              "event_name_1": event_name_1_input, 
              "event_type_1": event_type_1_input,
#              "event_name_2": event_name_2_input, 
              "event_type_2": event_type_2_input,
#               "item_id": item_id_input, 
              "dept_id": dept_id_input, "store_id": store_id_input, 
              "cat_id": cat_id_input, "state_id": state_id_input}

    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(loss=keras.losses.mean_squared_error,
                  metrics=["mse"],
                  optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


def create_model17EN1EN2emb1(num_dense_features, lr=0.002):
    tf.keras.backend.clear_session()
    gc.collect()

    # Dense input
    dense_input = Input(shape=(num_dense_features, ), name='dense1')

    # Embedding input
#     moonphase_input = Input(shape=(1,), name='moonphase')
    wday_input = Input(shape=(1,), name='wday')
#     month_input = Input(shape=(1,), name='month')
#     year_input = Input(shape=(1,), name='year')
    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
#    item_id_input = Input(shape=(1,), name='item_id')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')

    
    wday_emb = Flatten()(Embedding(7, 2)(wday_input))
#     month_emb = Flatten()(Embedding(12, 2)(month_input))
#     year_emb = Flatten()(Embedding(6, 2)(year_input))
    event_name_1_emb = Flatten()(Embedding(31, 1)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, 1)(event_type_1_input))
    event_name_2_emb = Flatten()(Embedding(5, 1)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, 1)(event_type_2_input))

#    item_id_emb = Flatten()(Embedding(3049, 6)(item_id_input))
    dept_id_emb = Flatten()(Embedding(7, 2)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 3)(store_id_input))
    cat_id_emb = Flatten()(Embedding(3, 2)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 2)(state_id_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    x = concatenate([dense_input, 
                     wday_emb, #month_emb, year_emb, 
                     event_name_1_emb, 
                     event_type_1_emb, 
                     event_name_2_emb, 
                     event_type_2_emb, 
#                      item_id_emb, 
                     dept_id_emb, store_id_emb,
                     cat_id_emb, state_id_emb])
    

    x = Dense(256*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256*2, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16*2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(4*2, activation="relu")(x)
    x = BatchNormalization()(x)
    
    outputs = Dense(1, activation="linear", name='output')(x)

    inputs = {"dense1": dense_input,
              "wday": wday_input, 
#               "month": month_input, "year": year_input, 
              "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,
              "event_name_2": event_name_2_input, "event_type_2": event_type_2_input,
#               "item_id": item_id_input, 
              "dept_id": dept_id_input, "store_id": store_id_input, 
              "cat_id": cat_id_input, "state_id": state_id_input}

    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(loss=keras.losses.mean_squared_error,
                  metrics=["mse"],
                  optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model