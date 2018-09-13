import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from operator import add

soma = []
soma1 = []
soma2 = []
soma3 = []


def distancia_euclidiana(vet1, vet2):
    vet1, vet2 = np.array(vet1), np.array(vet2)
    diff = vet1 - vet2
    quad_dist = np.dot(diff, diff)
    return quad_dist


def classifica_inversa(vetor, k, condi):
    vetor.sort(key=lambda tup: tup[0])
    vetor = vetor[0:5]

    if condi is 1:
        classes = np.array([0, 0])
        for x in vetor:
            classes[int(x[1]) - 1] += 1 / x[0]

        return np.argmax(classes) + 1
    else:
        classes = {}

        for x in vetor:
            if x[1] in classes:
                classes[x[1]] += 1
            else:
                classes[x[1]] = 1

        ans = max(classes, key=classes.get)

        return ans


def classifica_normalizada(vetor, k, condi):
    vetor.sort(key=lambda tup: tup[0])
    vetor = vetor[0:5]

    dists = list(map(lambda x: x[0], vetor))
    valor_maximo = np.amax(dists)
    valor_minimo = np.amin(dists)
    dists = (dists - valor_minimo) / (valor_maximo - valor_minimo)
    dists = np.subtract(1, dists)

    if condi is 1:
        classes = np.array([0, 0])

        for index, x in enumerate(vetor):
            classes[int(x[1]) - 1] += dists[index]

        return np.argmax(classes) + 1
    else:
        classes = {}

        for x in vetor:
            if x[1] in classes:
                classes[x[1]] += 1
            else:
                classes[x[1]] = 1

        ans = max(classes, key=classes.get)

        return ans


def verifica_classe(respostas, teste):
    acertos = 0
    erros = 0
    for index, i in enumerate(respostas):
        df = teste.iloc[index]
        if df['Class'] == i:
            acertos += 1
        else:
            erros += 1
    acuracia = acertos / 400
    taxa_erro = erros / 400
    print("Acurácia = ", acuracia, "e Taxa de Erro = ", taxa_erro)

    return acuracia


def knn(teste, treino, k):
    voto_majoritario_inversa = []
    voto_acumulado_normalizada = []
    voto_acumulado_inversa = []
    voto_majoritario_normalizada = []

    for x, y in teste.iterrows():
        vet_distancia_invesra = []
        vet_distancia_normalizadas = []
        v1 = [y['A1'], y['A2']]

        for X, Y in treino.iterrows():
            v2 = [Y['A1'], Y['A2']]

            vet_distancia_invesra.append([distancia_euclidiana(v1, v2), Y['Class']])
            vet_distancia_normalizadas.append([distancia_euclidiana(v1, v2), Y['Class']])

        voto_acumulado_inversa.append(classifica_inversa(vet_distancia_invesra, k, 1))
        voto_majoritario_inversa.append(classifica_inversa(vet_distancia_invesra, k, 2))
        voto_acumulado_normalizada.append(classifica_normalizada(vet_distancia_normalizadas, k, 1))
        voto_majoritario_normalizada.append(classifica_normalizada(vet_distancia_normalizadas, k, 2))

    #voto acumulado com a inversa da euclidiana
    val = verifica_classe(voto_acumulado_inversa, teste)
    #voto majoritario com a inversa da euclidiana
    val1 = verifica_classe(voto_majoritario_inversa, teste)
    #voto  acumulado com a normalizada da euclidiana
    val2 = verifica_classe(voto_acumulado_normalizada, teste)
    #voto majoritario com a normalizada da euclidiana
    val3 = verifica_classe(voto_majoritario_normalizada, teste)
    return val, val1, val2, val3


def getFolds(data_set):
    kf = KFold(n_splits=5, shuffle=True)
    return kf.split(data_set)


# lê o CSV
data_frame = pd.read_csv("Banana.csv", sep=',')

# quantidade de linhas do data frame
linhas = data_frame.shape[0]
colunas = data_frame.shape[1]

# pega todos os atributos
X = data_frame.iloc[0:linhas, 0:colunas - 1]

# pega todos os index do arquivo
Y = data_frame.iloc[:, -1]
# para o KNN
k = 5

media_fold = [0, 0, 0, 0]


for index in range(0, 10):
    kfold = getFolds(data_frame)
    resultado = [0, 0, 0, 0]
    for i, fold in enumerate(kfold):
        print("\nFold ", i)
        X_train, X_test = X.iloc[fold[0]], X.iloc[fold[1]]
        Y_train, Y_test = Y.iloc[fold[0]], Y.iloc[fold[1]]
        treino = X_train.join(Y_train)
        teste = X_test.join(Y_test)
        resultado = list(map(add, resultado, knn(teste, treino, k)))

    resultado = np.array(resultado)
    media_fold = resultado / 5

    with open('soma.csv', 'a') as filehandle:
        for listitem in media_fold:
            filehandle.write('%s, ' % listitem)
        filehandle.write("\n")

