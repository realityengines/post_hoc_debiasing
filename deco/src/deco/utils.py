import numpy as np

from sklearn.metrics import accuracy_score

from keras.layers import Input, Dense, Dropout
from keras.models import Model
import keras
from keras.layers import Input, Dense, BatchNormalization


def create_model_nn(num_hidden_layers, input_layer_width, hidden_layers_width, input_features_num):
    inputs = Input(shape=(input_features_num,))

    x = Dense(input_layer_width, activation='relu')(inputs)
    #x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)

    for i in range(num_hidden_layers):
        x = Dense(hidden_layers_width, activation='relu')(x)
        #x = Dropout(rate=0.2)(x)
        x = BatchNormalization()(x)

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def clip_at_threshold(y, threshold=0.5):
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


def accuracy_from_scores(y_true, y_score, threshold):
    y_pred = np.copy(y_score)
    y_pred[y_pred <= threshold] = 0
    y_pred[y_pred > threshold] = 1
    return accuracy_score(y_true, y_pred)

#### Original source in ./posthoc.
#### Adding here, so importing is easier in this level.
def zero_if_nan(x):
    return 0. if np.isnan(x) else x

def compute_bias(y_pred, y_true, priv, metric):
    gtpr_priv = zero_if_nan(y_pred[priv * y_true == 1].mean())
    gfpr_priv = zero_if_nan(y_pred[priv * (1-y_true) == 1].mean())
    mean_priv = zero_if_nan(y_pred[priv == 1].mean())

    gtpr_unpriv = zero_if_nan(y_pred[(1-priv) * y_true == 1].mean())
    gfpr_unpriv = zero_if_nan(y_pred[(1-priv) * (1-y_true) == 1].mean())
    mean_unpriv = zero_if_nan(y_pred[(1-priv) == 1].mean())

    if metric == "statistical_parity_difference":
        return mean_unpriv - mean_priv
    elif metric == "average_abs_odds_difference":
        return 0.5 * ((gfpr_unpriv - gfpr_priv) + (gtpr_unpriv - gtpr_priv))
    elif metric == "equal_opportunity_difference":
        return gtpr_unpriv - gtpr_priv
