import sklearn.metrics as metrics


f = open('./graph_exp1_new.csv', 'r')
data = f.readlines()
d = data[8][:-1].split(',')
d2 = data[9][:-1].split(',')
prob = []
y_true = []
for i in d:
    prob.append(5-int(i))
for i in d2:
    y_true.append(int(i))
t_auc = metrics.roc_auc_score(y_true, prob)
print(t_auc)
