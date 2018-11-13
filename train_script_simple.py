from build_dataset import readDataFromCsv
import tensorflow as tf

data_filename = "data/processed_data/data_set_2018-11-10 15_21_47"
x_train, y_train, x_test, y_test = readDataFromCsv(data_filename)

x_train = x_train.transpose()
y_train = y_train.transpose()
x_test = x_test.transpose()
y_test = y_test.transpose()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, input_dim=x_train.shape[1]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, shuffle=True)

accuracy = model.evaluate(x_test, y_test, verbose=1)
print("test loss = " + str(accuracy[0]) +", accuracy = " + str(accuracy[1]))

model.predict()