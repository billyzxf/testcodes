import os
import pickle

import tensorflow as tf


max_len_dict = {
    'called_products_0009': 10,
    'called_products_1019': 6,
    'called_products_date_0009': 10,
    'called_products_date_1019': 6,
    'clicked_products_0009': 3,
    'clicked_products_date_0009': 3,
    'host_0009': 10,
    'host_1019': 10,
    'host_2029': 10,
    'host_date_0009': 10,
    'host_date_1019': 10,
    'host_date_2029': 10,
    'host_fre_0009': 10,
    'host_fre_1019': 10,
    'host_fre_2029': 10,
    'keypress_120': 3,
    'keypress_30': 3,
    'keypress_60': 3,
    'keypress_90': 3,
    'label_0009': 3,
    'label_1019': 3,
    'label_date_0009': 3,
    'label_date_1019': 3,
    'model_value': 3,
    'outbound_sent_products_0009': 3,
    'outbound_sent_products_date_0009': 3,
    'picked_products_0009': 3,
    'picked_products_date_0009': 3,
    'rule_name_120': 3,
    'rule_name_30': 3,
    'rule_name_60': 3,
    'rule_name_90': 3,
    'semantic_120': 5,
    'semantic_30': 3,
    'semantic_60': 3,
    'semantic_90': 5,
    'set_all_ins_host_180': 5,
    'set_all_ins_host_360': 4,
    'sms_sent_products_0009': 10,
    'sms_sent_products_1019': 9,
    'sms_sent_products_date_0009': 10,
    'sms_sent_products_date_1019': 9
 }



def get_text_feature_type(col):
    if 'set_' in col:
        text_feature_type = 'insurances'
    elif 'host_' in col:
        text_feature_type = 'hosts'
    elif 'products_' in col:
        text_feature_type = 'products'
    elif 'label_' in col:
        text_feature_type = 'labels'
    elif 'keypress_' in col:
        text_feature_type = 'keypress'
    elif 'rule_name_' in col:
        text_feature_type = 'rules'
    elif 'semantic_' in col:
        text_feature_type = 'semantics'
    else:
        text_feature_type = 'model_value'
    return text_feature_type


max_processes = 20
oov_tok = '<OOV>'
vocab_size_dict = {col: vocab_info[1]+100 for col, vocab_info in vocabulary_dict.items()}


def get_token(text_feature_type):
    print(text_feature_type)
    token = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size_dict[text_feature_type], oov_token=oov_tok,
                                                  filters='')
    token.fit_on_texts(train_df[text_feature_type])
    return {text_feature_type: token}
	
	



token_dict = {}
text_feature_types = ['hosts', 'products', 'labels', 'keypress', 'rules', 'semantics', 'insurances', 'model_value']
with Pool(min(max_processes, len(text_feature_types))) as p:
    result_dict = {col: value for result_dict in p.map(get_token, text_feature_types) for col, value in
                   result_dict.items()}

for text_feature_type in text_feature_types:
    token_dict[text_feature_type] = result_dict[text_feature_type]

os.system(f'mkdir -p /home/zhaoxiangfei/model_dl/{data_code}/dicts/')
with open(f'/home/zhaoxiangfei/model_dl/{data_code}/dicts/token_dict.pkl', 'wb') as f:
    pickle.dump(token_dict, f)

# release memory
train_df = train_df.drop(columns=['hosts', 'products', 'labels', 'keypress', 'rules', 'semantics', 'insurances'])
test_df = test_df.drop(columns=['hosts', 'products', 'labels', 'keypress', 'rules', 'semantics', 'insurances'])

# tokenization and padding
padding_type = 'post'
truncate_type = 'post'


def tokenize(col):
    print(col)
    text_feature_type = get_text_feature_type(col)
    token = token_dict[text_feature_type]
    max_len = max_len_dict[col]
    tokenized_seq = token.texts_to_sequences(train_df[col])
    result_train = tf.keras.preprocessing.sequence.pad_sequences(tokenized_seq, maxlen=max_len, padding=padding_type,
                                                                 truncating=truncate_type)
    tokenized_seq = token.texts_to_sequences(test_df[col])
    result_test = tf.keras.preprocessing.sequence.pad_sequences(tokenized_seq, maxlen=max_len, padding=padding_type,
                                                                truncating=truncate_type)
    os.system(f'mkdir -p /home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/train')
    os.system(f'mkdir -p /home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/test')
    with open(f'/home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/train/{col}.pkl', 'wb') as f:
        pickle.dump(result_train, f)
    with open(f'/home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/test/{col}.pkl', 'wb') as f:
        pickle.dump(result_test, f)


os.system(f'rm -r /home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/*')
with Pool(min(max_processes, len(text_features))) as p:
    p.map(tokenize, text_features)


# Y train
y_train_cro = np.array(train_df['label'].tolist())

# Y test
y_test_cro = np.array(test_df['label'].tolist())

# X
X_train = {'numerical_features': np.array(train_df['numerical_features'].tolist())}
X_test = {'numerical_features': np.array(test_df['numerical_features'].tolist())}

for col in categorical_features:
    X_train[col] = np.array(train_df[col])
    X_test[col] = np.array(test_df[col])


# load text features to data dict
for col in text_features:
    with open(f'/home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/train/{col}.pkl', 'rb') as f:
        X_train[col] = pickle.load(f)
    with open(f'/home/zhaoxiangfei/model_dl/{data_code}/dataset/text_features/test/{col}.pkl', 'rb') as f:
        X_test[col] = pickle.load(f)


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, factor_order, activation=None, linear_regularizer=None, factor_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.factor_order = factor_order
        self.activation = tf.keras.activations.get(activation)
        self.linear_regularizer = tf.keras.regularizers.get(linear_regularizer)
        self.factor_regularizer = tf.keras.regularizers.get(factor_regularizer)
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.keras.backend.floatx(), shape=(None, input_dim))

        self.b = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        self.w = self.add_weight(name='one', shape=(input_dim, self.output_dim), initializer='glorot_uniform',
                                 trainable=True, regularizer=self.linear_regularizer)
        self.v = self.add_weight(name='two', shape=(input_dim, self.factor_order), initializer='glorot_uniform',
                                 trainable=True, regularizer=self.factor_regularizer)

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X_square = tf.keras.backend.square(inputs)

        xv = tf.keras.backend.square(tf.keras.backend.dot(inputs, self.v))
        xw = tf.keras.backend.dot(inputs, self.w)

        p = 0.5 * tf.keras.backend.sum(xv - tf.keras.backend.dot(X_square, tf.keras.backend.square(self.v)), 1)
        rp = tf.keras.backend.repeat_elements(tf.keras.backend.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = tf.keras.backend.reshape(f, (-1, self.output_dim))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim


def DeepFM(output_units=1, dnn_hidden_units=[200, 100, 100, 50],
           dropout_rate_dnn_input=0.1, dropout_rate_dnn_logit=0.1, l2_reg_dnn=0.0,
           fm_hidden_units=[100, 80, 50], fm_factor_order=2, dropout_rate_fm_input=0.1, dropout_rate_fm_logit=0.1,
           l2_reg_fm_linear=0.0, l2_reg_fm_factor=0.0,
           l2_reg_text_embedding=0.0, l2_reg_categorical_embedding=0.0, l2_reg_numerical_embedding=0.0):
    tf.keras.backend.clear_session()

    # text inputs
    text_interactions = [
        ['host_0009', 'host_date_0009', 'host_fre_0009'],
        ['host_1019', 'host_date_1019', 'host_fre_1019'],
        ['host_2029', 'host_date_2029', 'host_fre_2029'],
        ['label_0009', 'label_date_0009'],
        ['label_1019', 'label_date_1019'],
        ['clicked_products_0009', 'clicked_products_date_0009'],
        ['sms_sent_products_0009', 'sms_sent_products_date_0009'],
        ['sms_sent_products_1019', 'sms_sent_products_date_1019'],
        ['called_products_0009', 'called_products_date_0009'],
        ['called_products_1019', 'called_products_date_1019'],
        ['picked_products_0009', 'picked_products_date_0009'],
        ['outbound_sent_products_0009', 'outbound_sent_products_date_0009'],
    ]
    text_inputs = [tf.keras.layers.Input(shape=max_len_dict[col], name=col) for col in text_features]
    text_embeddings = []
    for i, col in enumerate(text_features):
        text_feature_type = get_text_feature_type(col)
        text_embeddings.append(
            tf.keras.layers.Embedding(vocab_size_dict[text_feature_type],
                                      int(np.log1p(vocab_size_dict[text_feature_type]) + 1),
                                      embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_text_embedding),
                                      name=col + '_embed')(text_inputs[i])
        )

    text_multiplys = []
    for text_interaction in text_interactions:
        text_multiplys.append(
            tf.keras.layers.Multiply(name=text_interaction[0] + '_multiply')(
                [text_embeddings[text_features.index(ele)] for ele in text_interaction])
        )

    text_logit = tf.keras.layers.Concatenate(name='text_concat')(
        [tf.keras.layers.GlobalAveragePooling1D()(text_mul) for text_mul in text_multiplys] +
        [tf.keras.layers.GlobalAveragePooling1D()(text_emb) for text_emb in text_embeddings]
    )

    # categorical inputs
    categorical_inputs = [tf.keras.layers.Input(shape=(1,), name=col) for col in categorical_features]
    categorical_embeddings = []
    for i, col in enumerate(categorical_features):
        categorical_embeddings.append(
            tf.keras.layers.Embedding(category_counts_dict[col], int(np.log1p(category_counts_dict[col]) + 1),
                                      embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_categorical_embedding),
                                      name=col + '_embed')(categorical_inputs[i])
        )

    categorical_logit = tf.keras.layers.Concatenate(name='categorical_concat')(
        [tf.keras.layers.Flatten()(cat_emb) for cat_emb in categorical_embeddings]
    )

    # numerical inputs
    numerical_input = tf.keras.layers.Input(shape=[len(numerical_features)], name='numerical_features')

    # dnn
    dnn_input = tf.keras.layers.Concatenate(name='deep_concat')([text_logit, categorical_logit, numerical_input])
    dnn_logit = tf.keras.layers.Dropout(dropout_rate_dnn_input)(dnn_input)
    for n_unit in dnn_hidden_units:
        dnn_logit = tf.keras.layers.Dense(n_unit, activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn))(dnn_logit)
        # dnn_logit = tf.keras.layers.BatchNormalization()(dnn_logit)
        dnn_logit = tf.keras.layers.Dropout(dropout_rate_dnn_logit)(dnn_logit)

    # fm
    fm_input = tf.keras.layers.Concatenate(name='fm_concat')([text_logit, categorical_logit, numerical_input])
    fm_logit = tf.keras.layers.Dropout(dropout_rate_fm_input)(fm_input)
    for n_units in fm_hidden_units:
        fm_logit = FMLayer(n_units, fm_factor_order, activation='relu',
                           linear_regularizer=tf.keras.regularizers.l2(l2_reg_fm_linear),
                           factor_regularizer=tf.keras.regularizers.l2(l2_reg_fm_factor))(fm_logit)
        # fm_logit = tf.keras.layers.BatchNormalization()(fm_logit)
        fm_logit = tf.keras.layers.Dropout(dropout_rate_fm_logit)(fm_logit)

    outputs = tf.keras.layers.Concatenate()([dnn_logit, fm_logit])
    outputs = tf.keras.layers.Dense(output_units, activation='sigmoid')(outputs)

    model = tf.keras.models.Model(inputs=text_inputs + categorical_inputs + [numerical_input], outputs=outputs,
                                  name='DeepFM')
    return model


lr = 0.0025
n_epochs = 100
batch_size = 8192

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=1, min_lr=0.0001, verbose=1)
callbacks = [es, reduce_lr]


def train_model(model_type, x_train, y_train, x_test, y_test):
    output_units = 3
    model_code2 = model_code.replace('cro_sent', model_type)
    model = DeepFM(output_units)
    model.compile(
        loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO, alpha=0.5, gamma=3),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'), tf.keras.metrics.AUC(name='auc')],
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    print(f'{now()} Runtime Info: Started training {model_code2}:')
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=n_epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    model.save(f'/home/zhaoxiangfei/model_dl/{data_code}/{model_code2}/model/')
    print(f'{now()} Runtime Info: Finished saving {model_code2}')
    print(
        f'{now()} Runtime Info: --------------------------------------------------------------------------------------')