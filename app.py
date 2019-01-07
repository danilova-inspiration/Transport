from sklearn.ensemble import RandomForestClassifier
import csv
import codecs
import numpy as np
import time
from datetime import datetime as dt


# start time
start = time.time()

# classifier
clf = RandomForestClassifier(n_estimators = 155, max_features = None, criterion = 'entropy', oob_score = True)

# parse transport data
data = []
target = []
test = []

parsing_data = list(csv.reader(codecs.open('transport_data.csv', 'r', 'utf_8_sig')))

for i in range(1, len(parsing_data)):
    if parsing_data[i][4] == "?":
        dt_server_time = dt.fromtimestamp(int(parsing_data[i][2]))
        server_time = float(dt_server_time.hour) + float(dt_server_time.minute)/60. + float(dt_server_time.second)/3600.
        dt_transport_time = dt.fromtimestamp(int(parsing_data[i][3]))
        transport_time = float(dt_transport_time.hour) + float(dt_transport_time.minute)/60. + float(dt_transport_time.second)/3600.
        test.append([parsing_data[i][0], parsing_data[i][1], server_time, transport_time])
    elif parsing_data[i][4] != "-":
        dt_server_time = dt.fromtimestamp(int(parsing_data[i][2]))
        server_time = float(dt_server_time.hour) + float(dt_server_time.minute)/60. + float(dt_server_time.second)/3600.
        dt_transport_time = dt.fromtimestamp(int(parsing_data[i][3]))
        transport_time = float(dt_transport_time.hour) + float(dt_transport_time.minute)/60. + float(dt_transport_time.second)/3600.
        data.append([parsing_data[i][0], parsing_data[i][1], server_time, transport_time])
        target.append(parsing_data[i][4])

# end parse transport data

# train
clf.fit(data, target)

# predict
prediction = clf.predict(test)

np.savetxt('transport_output.txt', prediction, fmt="%s")

# end time
print((time.time() - start))
