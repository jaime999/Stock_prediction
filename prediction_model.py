import pandas as pd
from prepare_data import get_economic_data, get_generic_sentiment
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, SimpleRNN, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pydruid.db import connect
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt


LOOKBACK = 3

def get_articles(druid=False):
    if druid:
        # Conectar a Druid
        conn = connect(host='localhost', port=8082,
                        path='/druid/v2/sql/', scheme='http')

        query = "SELECT * FROM news_datasource"

        articles = pd.read_sql(query, conn)

        # # Cerrar la conexión
        # cursor.close()
        conn.close()
    
    else:
        articles = pd.read_csv('new_headlines_sentiment.csv')
        
    return articles
        

def get_data_filter_date(df, date='2024-01-01'):
    return df[df['Fecha'] >= date]

def group_articles_sentiment(articles):
    return articles.groupby('Fecha', as_index=False)[[
        'xlmr_positive', 'xlmr_neutral', 'xlmr_negative',
        'finance-sentiment-es_positive', 'finance-sentiment-es_neutral',
        'finance-sentiment-es_negative', 'mbert_oneStars',
        'mbert_twoStars', 'mbert_threeStars', 'mbert_fourStars',
        'mbert_fiveStars', 'finbert_positive', 'finbert_neutral',
        'finbert_negative', 'pysentimiento_positive',
        'pysentimiento_neutral', 'pysentimiento_negative']].mean()

def get_features(df, columnsToGet=['Apertura_bbva', 'Máximo_bbva', 'Mínimo_bbva',
       'RSI_bbva', 'Apertura_sabadell', 'Máximo_sabadell',
       'Mínimo_sabadell', 'RSI_sabadell', 'xlmr_positive', 'xlmr_neutral',
       'xlmr_negative']):
    return df[columnsToGet]

def merge_df(leftDf, rightDf, columnToMerge='Fecha', directionToMerge='left'):
    if columnToMerge == 'Fecha':
        leftDf['Fecha'] = pd.to_datetime(leftDf['Fecha'])
        rightDf['Fecha'] = pd.to_datetime(rightDf['Fecha'])
        
    return pd.merge(
        leftDf, rightDf, on=columnToMerge, how=directionToMerge)


def minMaxScaler():
    scalerFeatures = MinMaxScaler(feature_range=(0, 1))
    scalerTarget = MinMaxScaler(feature_range=(0, 1))

    return scalerFeatures, scalerTarget


def robustScaler():
    scalerFeatures = RobustScaler()
    scalerTarget = RobustScaler()

    return scalerFeatures, scalerTarget


def createDataset(features, target, lookBack):
    dataX, dataY = [], []
    for i in range(len(features) - lookBack):
        dataX.append(features[i:(i + lookBack), :])
        dataY.append(target[i + lookBack, :])

    return np.array(dataX), np.array(dataY)


class BaseModel(kt.HyperModel):
    def build(self, hp):
        raise NotImplementedError("Subclass must implement build method")

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32, 64]),
            **kwargs,
        )


class MyHyperModelRNN(BaseModel):
    def build(self, hp):
        model = Sequential()
        num_rnn_layers = hp.Int('num_rnn_layers', 1, 3)

        for i in range(num_rnn_layers):
            return_sequences = True if i < num_rnn_layers - 1 else False
            model.add(SimpleRNN(
                units=hp.Int(f'units_rnn_{i}', min_value=32,
                             max_value=512, step=32),
                activation=hp.Choice(f'activation_rnn_{i}', ['relu', 'tanh']),
                return_sequences=return_sequences
            ))
            model.add(Dropout(rate=hp.Float(
                f'dropout_rnn_{i}', min_value=0.0, max_value=0.5, step=0.1)))

            # Hiperparámetro para el número de capas Dense
        num_dense_layers = hp.Int('num_dense_layers', 0, 2)

        for i in range(num_dense_layers):
            model.add(Dense(
                units=hp.Int(f'dense_units_{i}',
                             min_value=32, max_value=512, step=32),
                activation=hp.Choice(f'activation_dense_{i}', ['relu', 'tanh']),
            ))
            # Añadir Dropout después de cada capa Dense
            model.add(Dropout(
                rate=hp.Float(f'dropout_dense_{i}',
                              min_value=0.0, max_value=0.5, step=0.1)
            ))

        model.add(Dense(2))  # Ajusta según tu problema

        model.compile(
            optimizer=Adam(hp.Choice('learning_rate',
                           values=[1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error',
            metrics=['mae']  # Métrica adecuada para regresión
        )

        return model


class MyHyperModelCNN(BaseModel):
    def build(self, hp):
        model = Sequential()
        # Hiperparámetro para el número de capas Conv1D
        num_conv_layers = hp.Int('num_conv_layers', 1, 3)

        for i in range(num_conv_layers):
            model.add(Conv1D(
                filters=hp.Int(f'filters_{i}', min_value=32,
                               max_value=128, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[2, 3]),
                activation=hp.Choice(f'activation_conv_{i}', ['relu', 'tanh']),
                padding="same"
            ))
            model.add(MaxPooling1D(2, padding='same'))
            # Añadir Dropout después de cada capa Conv1D
            model.add(Dropout(
                rate=hp.Float(f'dropout_conv_{i}',
                              min_value=0.0, max_value=0.5, step=0.1)
            ))

        model.add(Flatten())

        # Hiperparámetro para el número de capas Dense
        num_dense_layers = hp.Int('num_dense_layers', 0, 2)

        for i in range(num_dense_layers):
            model.add(Dense(
                units=hp.Int(f'dense_units_{i}',
                             min_value=32, max_value=512, step=32),
                activation=hp.Choice(f'activation_dense_{i}', ['relu', 'tanh']),
            ))
            # Añadir Dropout después de cada capa Dense
            model.add(Dropout(
                rate=hp.Float(f'dropout_dense_{i}',
                              min_value=0.0, max_value=0.5, step=0.1)
            ))

        model.add(Dense(2))  # Capa de salida
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate',
                           values=[1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error',
            metrics=['mae']  # Métrica adecuada para regresión
        )

        return model


class MyHyperModelLSTM(BaseModel):
    def build(self, hp):
        model = Sequential()
        # Hiperparámetro para el número de capas LSTM
        num_lstm_layers = hp.Int('num_lstm_layers', 1, 3)

        for i in range(num_lstm_layers):
            return_sequences = True if i < num_lstm_layers - 1 else False
            model.add(LSTM(
                units=hp.Int(f'units_lstm_{i}', min_value=32,
                             max_value=512, step=32),
                return_sequences=return_sequences,
                activation=hp.Choice(f'activation_lstm_{i}', ['relu', 'tanh'])
            ))
            model.add(Dropout(
                rate=hp.Float(
                    f'dropout_lstm_{i}', min_value=0.0, max_value=0.5, step=0.1)
            ))

        num_dense_layers = hp.Int('num_dense_layers', 0, 2)

        for i in range(num_dense_layers):
            model.add(Dense(
                units=hp.Int(f'dense_units_{i}',
                             min_value=32, max_value=512, step=32),
                activation=hp.Choice(f'activation_dense_{i}', ['relu', 'tanh']),
            ))
            # Añadir Dropout después de cada capa Dense
            model.add(Dropout(
                rate=hp.Float(f'dropout_dense_{i}',
                              min_value=0.0, max_value=0.5, step=0.1)
            ))

        model.add(Dense(2))
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate',
                           values=[1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model


def create_validation(x, y, val_size=0.2):
    n = len(x)
    split_at = int(n * (1 - val_size))

    return (
        x[:split_at],      # x_train
        x[split_at:],      # x_val
        y[:split_at],      # y_train
        y[split_at:]       # y_val
    )


def fineTunning(model_name, project_name):
    n_total = len(features)
    # Por ejemplo, el 20% de los datos para test
    test_size = int(0.2 * n_total)

    # El conjunto de test se toma de la parte final de la serie (información futura)
    data_train = features[:-test_size]
    targets_train = target[:-test_size]
    data_test = features[-test_size:]
    targets_test = target[-test_size:]

    scalerFeatures, scalerTarget = minMaxScaler()

    train_data_scaled = scalerFeatures.fit_transform(
        data_train)
    test_data_scaled = scalerFeatures.transform(data_test)
    train_targets_scaled = scalerTarget.fit_transform(
        targets_train)
    test_targets_scaled = scalerTarget.transform(targets_test)
    X_train, y_train = createDataset(train_data_scaled,
                                     train_targets_scaled, LOOKBACK)
    X_test, y_test = createDataset(test_data_scaled,
                                   test_targets_scaled, LOOKBACK)

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = create_validation(
        X_train, y_train, val_size=0.2)

    hypermodel = MyHyperModelRNN()
    # hypermodel = MyHyperModelCNN()
    #hypermodel = MyHyperModelLSTM()

    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_loss',
        max_trials=50,
        directory='my_dir',
        project_name=project_name
    )

    checkpoint = ModelCheckpoint(
        # Archivo donde se guardará el modelo
        filepath=f'models/{model_name}.h5',
        monitor='val_loss',
        save_best_only=True,         # Solo guarda el modelo si es el mejor hasta el momento
        mode='min',
        verbose=1                    # Imprime información cuando guarda el modelo
    )

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    tuner.search(
        X_train_sub,
        y_train_sub,
        epochs=50,
        validation_data=(X_val_sub, y_val_sub),
        callbacks=callbacks
    )

    # al terminar la búsqueda:
    tuner.results_summary()     # Resumen de los mejores trials
    best_hps = tuner.get_best_hyperparameters()[0]
    print("Mejores hiperparámetros:", best_hps.values)

    # Si quieres reconstruir el mejoqr modelo y ajustarlo más:
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        X_train_sub, y_train_sub,
        epochs=100,
        validation_data=(X_val_sub, y_val_sub),
        callbacks=[checkpoint]
    )

    best_model = load_model(f'models/{model_name}.h5')
    y_pred = best_model.predict(X_test)
    trainPredict = best_model.predict(X_train)

    # Aplicar la transformación inversa a ambos resultados
    trainPredictInv = scalerTarget.inverse_transform(trainPredict)
    trainYInv = scalerTarget.inverse_transform(y_train)  # Asumiendo que y_train tiene dos columnas
    testPredictInv = scalerTarget.inverse_transform(y_pred)
    testYInv = scalerTarget.inverse_transform(y_test)  # Asumiendo que y_test tiene dos columnas

    mseTrainBBVA = mean_squared_error(
        trainYInv[:, 0], trainPredictInv[:, 0])
    mseTrainSabadell = mean_squared_error(
        trainYInv[:, 1], trainPredictInv[:, 1])
    print(f'MSE entrenamiento BBVA: {mseTrainBBVA:.5f}')
    print(f'MSE entrenamiento Banco Sabadell: {mseTrainSabadell:.5f}')

    rmseTrainBBVA = np.sqrt(mseTrainBBVA)
    rmseTrainSabadell = np.sqrt(mseTrainSabadell)
    print(f'RMSE entrenamiento BBVA: {rmseTrainBBVA:.5f}')
    print(f'RMSE entrenamiento Banco Sabadell: {rmseTrainSabadell:.5f}')

    maeTrainBBVA = mean_absolute_error(
        trainYInv[:, 0], trainPredictInv[:, 0])
    maeTrainSabadell = mean_absolute_error(
        trainYInv[:, 1], trainPredictInv[:, 1])
    print(f'MAE entrenamiento BBVA: {maeTrainBBVA:.5f}')
    print(f'MAE entrenamiento Banco Sabadell: {maeTrainSabadell:.5f}')

    mapeTrainBBVA = mean_absolute_percentage_error(
        trainYInv[:, 0], trainPredictInv[:, 0])
    mapeTrainSabadell = mean_absolute_percentage_error(
        trainYInv[:, 1], trainPredictInv[:, 1])
    print(f'MAPE entrenamiento BBVA: {mapeTrainBBVA:.5f}')
    print(f'MAPE entrenamiento Banco Sabadell: {mapeTrainSabadell:.5f}')
    
    mseTestBBVA = mean_squared_error(
        testYInv[:, 0], testPredictInv[:, 0])
    mseTestSabadell = mean_squared_error(
        testYInv[:, 1], testPredictInv[:, 1])
    print(
        f"MSE test BBVA: {mseTestBBVA:.5f}")
    print(
        f"MSE test Sabadell: {mseTestSabadell:.5f}")
    
    rmseTestBBVA = np.sqrt(mseTestBBVA)
    rmseTestSabadell = np.sqrt(mseTestSabadell)
    print(f'RMSE test BBVA: {rmseTestBBVA:.5f}')
    print(f'RMSE test Banco Sabadell: {rmseTestSabadell:.5f}')
    
    maeTestBBVA = mean_absolute_error(
        testYInv[:, 0], testPredictInv[:, 0])
    maeTestSabadell = mean_absolute_error(
        testYInv[:, 1], testPredictInv[:, 1])
    print(
        f"MAE test BBVA: {maeTestBBVA:.5f}")
    print(
        f"MAE test Sabadell: {maeTestSabadell:.5f}")

    mapeTestBBVA = mean_absolute_percentage_error(
        testYInv[:, 0], testPredictInv[:, 0])
    mapeTestSabadell = mean_absolute_percentage_error(
        testYInv[:, 1], testPredictInv[:, 1])
    print(
        f"MAPE test BBVA: {mapeTestBBVA:.5f}")
    print(
        f"MAPE test Sabadell: {mapeTestSabadell:.5f}")

    # Calcular el RMSE y el accuracy para cada resultado

    trainAccuracy1 = 100 - (rmseTrainBBVA / np.mean(trainYInv[:, 0]) * 100)
    trainAccuracy2 = 100 - (rmseTrainSabadell / np.mean(trainYInv[:, 1]) * 100)
    testAccuracy1 = 100 - (rmseTestBBVA / np.mean(testYInv[:, 0]) * 100)
    testAccuracy2 = 100 - (rmseTestSabadell / np.mean(testYInv[:, 1]) * 100)

    print(f'Precisión entrenamiento BBVA: {trainAccuracy1:.2f}%')
    print(f'Precisión entrenamiento Sabadell: {trainAccuracy2:.2f}%')
    print(f'Precisión test BBVA: {testAccuracy1:.2f}%')
    print(f'Precisión test Banco Sabadell: {testAccuracy2:.2f}%')

    # Crear gráficas para ambos resultados
    dates = dfMerged['Fecha'].values
    
    plt.figure(figsize=(20, 10))
    plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
             testYInv[:, 0], label='Test real BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testEconomicDataNormalizedRnnFiltered[:, 0], label='Test predicción datos normalizados RNN BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_pred)],
    #          testDataMergedFinanceSentimentEsLstm[:, 0],
    #          label='Test predicción Finance-sentiment-es LSTM BBVA')
    # plt.xlim(left=pd.to_datetime('2025-06-18'))
    # plt.xlim(right=pd.to_datetime('2025-10-30'))
    plt.ylim(bottom=12)  # Ajusta el límite inferior del eje y
    plt.legend()
    
    
    plt.figure(figsize=(20, 10))
    plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
             testYInv[:, 1], label='Test real Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testEconomicDataNormalizedRnnFiltered[:, 1],
    #          label='Test predicción datos normalizados RNN Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testDataMergedFinanceSentimentEsLstm[:, 1],
    #          label='Test predicción Finance-sentiment-es LSTM Banco Sabadell')
    # plt.xlim(left=pd.to_datetime('2025-06-18'))
    # plt.xlim(right=pd.to_datetime('2025-10-30'))
    plt.ylim(bottom=2.6)  # Ajusta el límite inferior del eje y
    plt.legend()
    
    
    # Gráfica para el primer resultado
    plt.figure(figsize=(20, 10))
    plt.plot(dates[:len(y_train)], trainYInv[:, 0],
             label='Entrenamiento real BBVA')
    # plt.plot(dates[:len(y_train)], trainEconomicDataNormalizedRNN[:, 0],
    #          label='Entrenamiento predicción datos normalizados RNN BBVA')
    # plt.plot(dates[:len(y_train)], trainEconomicDataRsiLstm[:, 0],
    #          label='Entrenamiento predicción RSI LSTM BBVA')
    # plt.plot(dates[:len(y_train)], trainEconomicDataCnn[:, 0],
    #          label='Entrenamiento predicción datos originales CNN BBVA')
    plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
             testYInv[:, 0], label='Test real BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testEconomicDataNormalizedRNN[:, 0], label='Test predicción datos normalizados RNN BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_pred)],
    #          testEconomicDataRsiLstm[:, 0], label='Test predicción RSI LSTM BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_pred)],
    #          testEconomicDataCnn[:, 0], label='Test predicción datos originales CNN BBVA')
    # plt.xlim(left=pd.to_datetime('2024-01-01'))
    # plt.xlim(right=pd.to_datetime('2025-10-30'))
    plt.ylim(bottom=6)  # Ajusta el límite inferior del eje y
    plt.legend()

    # Gráfica para el segundo resultado
    plt.figure(figsize=(20, 10))
    plt.plot(dates[:len(y_train)], trainYInv[:, 1],
             label='Entrenamiento real Banco Sabadell')
    # plt.plot(dates[:len(y_train)], trainEconomicDataNormalizedRNN[:, 1],
    #          label='Entrenamiento predicción datos normalizados RNN Banco Sabadell')
    # plt.plot(dates[:len(y_train)], trainEconomicDataRsiLstm[:, 1],
    #          label='Entrenamiento predicción RSI LSTM Banco Sabadell')
    # plt.plot(dates[:len(y_train)], trainEconomicDataCnn[:, 1],
    #          label='Entrenamiento predicción datos originales CNN Banco Sabadell')
    plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
             testYInv[:, 1], label='Test real Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testEconomicDataNormalizedRNN[:, 1], label='Test predicción datos normalizados RNN Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testEconomicDataRsiLstm[:, 1], label='Test predicción RSI LSTM Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testEconomicDataCnn[:, 1], label='Test predicción datos originales Banco Sabadell')
    # plt.xlim(left=pd.to_datetime('2024-01-01'))

    # plt.xlim(left=pd.to_datetime('2024-01-01'))
    # plt.xlim(right=pd.to_datetime('2025-10-30'))
    plt.ylim(bottom=1)  # Ajusta el límite inferior del eje y
    plt.legend()

    plt.figure(figsize=(20, 10))
    # plt.plot(dates[:len(y_train)], trainYInv[:, 0],
    #          label='Entrenamiento real BBVA')
    # plt.plot(dates[:len(y_train)], trainDataMergedXlmrCnn[:, 0],
    #          label='Entrenamiento predicción XLM-R CNN BBVA')
    # plt.plot(dates[:len(y_train)], trainDataMergedAllModelsCnn[:, 0],
    #          label='Entrenamiento predicción todos CNN BBVA')
    # plt.plot(dates[:len(y_train)], trainDataMergedFinanceSentimentEsLstm[:, 0],
    #          label='Entrenamiento predicción Finance-sentiment-es LSTM BBVA')
    plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
             testYInv[:, 0], label='Test real BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testDataMergedXlmrCnn[:, 0], label='Test predicción XLM-R CNN BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_pred)],
    #          testDataMergedAllModelsCnn[:, 0], label='Test  predicción todos CNN BBVA')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_pred)],
    #          testDataMergedFinanceSentimentEsLstm[:, 0],
    #          label='Test predicción Finance-sentiment-es LSTM BBVA')
    # plt.xlim(left=pd.to_datetime('2024-01-01'))
    # plt.xlim(right=pd.to_datetime('2025-10-30'))
    plt.ylim(bottom=6)  # Ajusta el límite inferior del eje y
    plt.legend()

    # Gráfica para el segundo resultado
    plt.figure(figsize=(20, 10))
    plt.plot(dates[:len(y_train)], trainYInv[:, 1],
             label='Entrenamiento real Banco Sabadell')
    # plt.plot(dates[:len(y_train)], trainDataMergedXlmrCnn[:, 1],
    #          label='Entrenamiento predicción XLM-R CNN Banco Sabadell')
    # plt.plot(dates[:len(y_train)], trainDataMergedAllModelsCnn[:, 1],
    #          label='Entrenamiento predicción todos CNN Banco Sabadell')
    # plt.plot(dates[:len(y_train)], trainDataMergedFinanceSentimentEsLstm[:, 1],
    #          label='Entrenamiento predicción Finance-sentiment-es LSTM Banco Sabadell')
    plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
             testYInv[:, 1], label='Test real Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testDataMergedXlmrCnn[:, 1],
    #          label='Test predicción XLM-R CNN Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testDataMergedAllModelsCnn[:, 1],
    #          label='Test predicción todos CNN Banco Sabadell')
    # plt.plot(dates[len(y_train):len(y_train) + len(y_test)],
    #          testDataMergedFinanceSentimentEsLstm[:, 1],
    #          label='Test predicción Finance-sentiment-es LSTM Banco Sabadell')
    # plt.xlim(left=pd.to_datetime('2024-01-01'))

    # plt.xlim(left=pd.to_datetime('2024-01-01'))
    # plt.xlim(right=pd.to_datetime('2025-10-30'))
    plt.ylim(bottom=1)  # Ajusta el límite inferior del eje y
    plt.legend()
    
    plt.show()

def crossValidation(features, target, scalerFeatures, scalerTarget):
    lookBack = 3

    data = features
    targets = target

    # Parámetros para la validación walk‑forward:
    n_splits = 5
    n_total = len(data)
    test_size = int(0.2 * n_total)

    data_train_val = data[:-test_size]
    n_train_val = len(data_train_val)
    val_size = n_train_val // (n_splits + 1)

    mse_scores = []
    print("=== Validación Walk-forward con Lookback ===")
    for i in range(1, n_splits + 1):
        train_end = i * val_size
        val_start = train_end
        val_end = val_start + val_size
        if val_end > n_total:
            break

        train_data = data[:train_end]
        train_targets = targets[:train_end]
        val_data = data[val_start:val_end]
        val_targets = targets[val_start:val_end]


        train_data_scaled = scalerFeatures.fit_transform(
            train_data)
        val_data_scaled = scalerFeatures.transform(val_data)
        train_targets_scaled = scalerTarget.fit_transform(
            train_targets.reshape(-1, 1))
        val_targets_scaled = scalerTarget.transform(val_targets.reshape(-1, 1))
        X_train_seq, y_train_seq = createDataset(train_data_scaled,
                                                 train_targets_scaled, lookBack)
        X_val_seq, y_val_seq = createDataset(val_data_scaled,
                                             val_targets_scaled, lookBack)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            print(
                f"Split {i}: No se generaron suficientes secuencias. Se requiere aumentar el tamaño del conjunto o reducir el lookback.")
            continue

        # X_train_seq = X_train_seq.reshape(-1, lookback, 1)
        # X_val_seq = X_val_seq.reshape(-1, lookback, 1)
        X_train_seq = np.reshape(
            X_train_seq, (X_train_seq.shape[0], lookBack, X_train_seq.shape[2]))
        X_val_seq = np.reshape(
            X_val_seq, (X_val_seq.shape[0], lookBack, X_val_seq.shape[2]))

        # # --- Entrenamiento y Evaluación del Modelo ---
        # input_shape = (lookback, 1)
        # model = build_model(lookBack)

        checkpoint = ModelCheckpoint(
            # Archivo donde se guardará el modelo
            filepath='models/best_lstm_model_crossValidation.h5',
            monitor='loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        # model.fit(X_train_seq, y_train_seq,
        #           epochs=50, batch_size=32, callbacks=[checkpoint])

        best_model = load_model('models/best_lstm_model_crossValidation.h5')
        y_pred_val = best_model.predict(X_val_seq)
        mse = mean_squared_error(y_val_seq, y_pred_val)
        mse_scores.append(mse)
        print(
            f"Split {i}: Entrenamiento hasta índice {train_end} y validación de {val_start} a {val_end} -> MSE: {mse:.4f}")

    meanScore = np.mean(mse_scores)
    print(f"\nMSE promedio en validación: {np.mean(mse_scores):.4f}")

    return meanScore


#dfMerged = merge_articles_economic(articles)

# compareAndTrainBestModel(features, target)
articles = get_articles()
articles = get_generic_sentiment(articles)
articles = group_articles_sentiment(articles)

dfBBVA = get_economic_data(
    'HistoricDataBBVA_01-01-20_31-10-25.csv', 'bbva')
dfSabadell = get_economic_data(
    'HistoricDataSabadell_01-01-20_31-10-25.csv', 'sabadell')
# dfBBVA = get_data_filter_date(dfBBVA)
# dfSabadell = get_data_filter_date(dfSabadell)

#dfMerged =  pd.merge(dfBBVA, dfSabadell, on='Fecha', how='inner')
dfMerged = merge_df(dfBBVA, dfSabadell)
dfMerged = merge_df(dfMerged, articles)
features = get_features(dfMerged)
target = dfMerged[['Último_bbva', 'Último_sabadell']].values
