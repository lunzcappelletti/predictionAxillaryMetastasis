# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score # suport K-fold
from sklearn.naive_bayes import MultinomialNB # Algoritimo Multinomial naive Bayes
from sklearn.ensemble import AdaBoostClassifier # Algoritimo Adaboost
from sklearn.multiclass import OneVsRestClassifier # Algoritimo OneVsRest
from sklearn.multiclass import OneVsOneClassifier # Algoritimo OneVsOne
from sklearn.svm import LinearSVC # Algoritimo suporte para OneVsRest e OneVsOne


# O objetivo dessa função é classificar um valor considerando a amplitude da classe
# Exemplo, Nós vamos transformar cala valor de uma coluna em binário, se considerarmos
# todos os valores entre 0 e 100, teremos 101 colunas binárias
# portanto, essa função com amplitude 5 e valor de 0 a 5, retorna 5, de 6 a 10, retorna 10
# e assim sucessivamente com base na amplitude
def classificacaoValor(valor, amplitude_classe, limite_superior):
    
    string = str(valor) #Função replace funciona apenas com strings
    #string_sem_ponto = string.replace(".",",") #Substitui pontos por virgula
    string_sem_espaco = string.replace(" ","") #Remove espaços
    numero_tratado = float(string_sem_espaco) #Converte para numero novamente

    i = 1
    while i < limite_superior:
        if numero_tratado <= i*amplitude_classe: return i*amplitude_classe       
        if i > limite_superior: break
        i += 1

# Importação do arquivo
caminho_arquivo_amostra = "C:\\Users\\Aluga.com\\Desktop\\DataScience\\amostra_cancer_mama.csv"
df = pd.read_csv(caminho_arquivo_amostra, sep=";")

# Tratativa de dados
# Cada uma das colunas do arquivo é separada em uma variavél, algumas delas (contém inteiros ou floats)
# recebem a função de classificacaoValor() para retornar a classificação

# A amplitude foi definida com base do desvio médio padrão (Foi feito isso manualmente

id_paciente = df['id']
sexo = df['sexo']
idade = df['idade'].apply(lambda x: classificacaoValor(x, 11, 120))
menopausa = df['menopausa']
historia_familiar = df['historia_familiar']
sindrome_hereditaria = df['sindrome_hereditaria']
lateralidade = df['lateralidade']
quadrante = df['quadrante']
tamanho_clinico_lesao = df['tamanho_clinico_lesao'].apply(lambda x: classificacaoValor(x, 10, 120))
palpabilidade = df['palpabilidade']
tipo_biopsia = df['tipo_biopsia']
laudo_anatomo_patologico = df['laudo_anatomo_patologico']
receptor_estrogenio_biopsia = df['receptor_estrogenio_biopsia'].apply(lambda x: classificacaoValor(x, 10, 120))
receptor_progesterona_biopsia = df['receptor_progesterona_biopsia'].apply(lambda x: classificacaoValor(x, 10, 120))
her2_biopsia = df['her2_biopsia']
ki67_biopsia = df['ki67_biopsia'].apply(lambda x: classificacaoValor(x, 10, 120))
data_cirurgia = df['data_cirurgia']
tipo_histologico_final = df['tipo_histologico_final']
tamanho_patologico_lesao = df['tamanho_patologico_lesao'].apply(lambda x: classificacaoValor(x, 10, 120))
multifocalidade = df['multifocalidade']
multicentricidade = df['multicentricidade']
grau_histologico = df['grau_histologico']
grau_nuclear = df['grau_nuclear']
invasao_vasculo_linfatica = df['invasao_vasculo_linfatica']
invasao_peridural = df['invasao_peridural']
invasao_vasculo_sanguinea = df['invasao_vasculo_sanguinea']
pesquisa_repetida = df['pesquisa_repetida']
receptor_estrogenio_final = df['receptor_estrogenio_final'].apply(lambda x: classificacaoValor(x, 10, 120))
receptor_progesterona_final = df['receptor_progesterona_final'].apply(lambda x: classificacaoValor(x, 10, 120))
ki67_final = df['ki67_final'].apply(lambda x: classificacaoValor(x, 10, 120))
her2_final = df['her2_final']
fish = df['fish']
indice_mitotico_final = df['indice_mitotico_final'].apply(lambda x: classificacaoValor(x, 10, 120))
necrose_final = df['necrose_final']
ca_in_situ_associado_final = df['ca_in_situ_associado_final'].apply(lambda x: classificacaoValor(x, 10, 120))
cirurgia_mama = df['cirurgia_mama']
cirurgia_axiliar = df['cirurgia_axiliar']
total_linfonodos = df['total_linfonodos'].apply(lambda x: classificacaoValor(x, 10, 120))
metastase_axiliar = df['metastase_axiliar']
linfonodos_comprometidos_mestastase = df['linfonodos_comprometidos_mestastase'].apply(lambda x: classificacaoValor(x, 2, 120))
tamanho_metastase_axiliar = df['tamanho_metastase_axiliar']#.apply(lambda x: agrupamento(x, 1, 20))
extravasamento_linfonodo_comprometido = df['extravasamento_linfonodo_comprometido']
tipo_metastase = df['tipo_metastase']



# Depois que cada coluna recebeu o tratamento vamos uni-las novamente, só que agora
# Iremos comentar as colunas que não fazem sentido para o algoritimo
# Esse processo foi feito observando a assertividade do algoritimo diante de cada 
# combinação de colunas com base na correlação de cada coluna com a metastase axilar

resultado_tratativa = pd.concat([
                        #id_paciente, 
                        #sexo,
                        idade,
                        #menopausa,
                        #historia_familiar,
                        sindrome_hereditaria,
                        #lateralidade,
                        quadrante,
                        tamanho_clinico_lesao,
                        palpabilidade,
                        tipo_biopsia,
                        laudo_anatomo_patologico,
                        receptor_estrogenio_biopsia,
                        receptor_progesterona_biopsia,
                        her2_biopsia,
                        ki67_biopsia,
                        tipo_histologico_final,
                        tamanho_patologico_lesao,
                        multifocalidade,
                        multicentricidade,
                        grau_histologico,
                        grau_nuclear,
                        invasao_vasculo_linfatica,
                        invasao_peridural,
                        invasao_vasculo_sanguinea,
                        #pesquisa_repetida,
                        receptor_estrogenio_final,
                        receptor_progesterona_final,
                        ki67_final,
                        her2_final,
                        fish,
                        indice_mitotico_final,
                        #necrose_final,
                        ca_in_situ_associado_final,
                        
						# Esses dados foram desconsiderados pois são resultados pós descoberta de metastase
						
						#cirurgia_mama,
                        #cirurgia_axiliar,
                        #total_linfonodos,
                        #metastase_axiliar,
                        #linfonodos_comprometidos_mestastase,
                        #tamanho_metastase_axiliar,
                        #extravasamento_linfonodo_comprometido,
                        #tipo_metastase
                   ], axis=1, sort=False)

# Separação de colunas para treino e marcações
X_df = resultado_tratativa
#X_df = df[['sexo','idade','menopausa','historia_familiar','sindrome_hereditaria','lateralidade','quadrante','tamanho_clinico_lesao','palpabilidade','tipo_biopsia','laudo_anatomo_patologico','receptor_estrogenio_biopsia','receptor_progesterona_biopsia','her2_biopsia','ki67_biopsia','data_cirurgia','tipo_histologico_final','tamanho_patologico_lesao','multifocalidade','multicentricidade','grau_histologico','grau_nuclear','invasao_vasculo_linfatica','invasao_peridural','invasao_vasculo_sanguinea','pesquisa_repetida','receptor_estrogenio_final','receptor_progesterona_final','ki67_final','her2_final','fish','indice_mitotico_final','necrose_final','ca_in_situ_associado_final','cirurgia_mama','cirurgia_axiliar','total_linfonodos']]
Y_df = df['metastase_axiliar']

# Transformação em Dummies
X = pd.get_dummies(X_df).fillna(0)
Y = pd.get_dummies(Y_df)['Sim - Neste caso, reportar-se às perguntas abaixo']

# Transformação em Array
X = X.values
Y = Y.values

# Separação de dados de treino, teste e validação
treino_dados = X[:300] 
treino_marcacao = Y[:300] 
k_fold = 10


multinomial = MultinomialNB()
adaboost = AdaBoostClassifier()
onevsrest = OneVsRestClassifier(LinearSVC(random_state = 0))
onevsone = OneVsOneClassifier(LinearSVC(random_state = 0))

def resultadoAlgoritimo(nome, modelo, treino_dados, treino_marcacao, cv = k_fold):
    scores = cross_val_score(modelo, treino_dados, treino_marcacao, cv = k_fold)
    media = np.mean(scores)
    print(nome, media)
    
resultadoAlgoritimo("multinomial:", multinomial, treino_dados, treino_marcacao, cv = k_fold)
resultadoAlgoritimo("adaboost:", adaboost, treino_dados, treino_marcacao, cv = k_fold)
resultadoAlgoritimo("onevsrest:", onevsrest, treino_dados, treino_marcacao, cv = k_fold)
resultadoAlgoritimo("onevsone:", onevsone, treino_dados, treino_marcacao, cv = k_fold)

target = [(np.count_nonzero(treino_marcacao)/len(treino_marcacao)), ((len(treino_marcacao)-np.count_nonzero(treino_marcacao))/len(treino_marcacao))]

print("target_minimo:", max(target))

modelo = AdaBoostClassifier()

teste = modelo.fit(treino_dados, treino_marcacao)
predicao = modelo.predict(X[82:])
diferencas = predicao - Y[82:]
acertos = [d for d in diferencas if d == 0]
print("teste Predição com algoritimo vencedor: ",len(acertos)  / len(Y[82:]))