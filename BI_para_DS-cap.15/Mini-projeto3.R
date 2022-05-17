# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=
#   Microsoft Power BI Para Data Science
#   MINI-PROJETO 3
#   PREVENDO A INADIMPLÊNCIA DE CLIENTES COM MACHINE LEARNING E POWER BI
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=

#SELECIONANDO PASTA DE TRABALHO
setwd("C:/Users/andre.alcantara/Documents/DSA/BI_para_DS-cap.15")
getwd()

########################INSTALL DE PACOTES
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=

########################CARREGANDO PACOTES
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=
library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=

########################CARREGANDO DADOS
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=
dados_clientes <- read.csv("dados/dataset.csv")

View(dados_clientes)
dim(dados_clientes)
summary(dados_clientes)
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=

########################ANALISE E LIMPEZA DOS DADOS
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=
#removendo a colina ID
dados_clientes$ID <- NULL
dim(dados_clientes)

#renomeando coluna classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)

#verificando se há valores vazios no dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main= 'Valores vazios')

#renomeando e convertendo genero, escolaridade, estado civil e idade 
#para categorias
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "genero"
colnames(dados_clientes)[3] <- "escolaridade"
colnames(dados_clientes)[4] <- "estado_civil"
colnames(dados_clientes)[5] <- "idade"
colnames(dados_clientes)

dados_clientes$genero <- cut(dados_clientes$genero,
                             c(0,1,2),
                             labels = c("Masculino",
                                        "Feminino"))

dados_clientes$escolaridade <- cut(dados_clientes$escolaridade,
                             c(0,1,2,3,4),
                             labels = c("Pós-graduação",
                                        "Graduação",
                                        "Ensino médio",
                                        "Outro"))

dados_clientes$estado_civil <- cut(dados_clientes$estado_civil,
                                   c(0,1,2,3),
                                   labels = c("Casado",
                                              "Solteiro",
                                              "Outro"))
#Categorizando idade para 0-30 anos= jovem, 31-50 anos= adulto, <50= idoso
dados_clientes$idade <- cut(dados_clientes$idade,
                            c(0,30,50,100),
                            labels = c("jovem",
                                       "adulto",
                                       "idoso"))

#Convertendo variaveis que indicam pagamento para fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#alterando variavel dependente para tipo fator
dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente)
str(dados_clientes$inadimplente)

#dataset pós conversões
View(dados_clientes)
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main="valores vazios")
dados_clientes <- na.omit(dados_clientes)
dim(dados_clientes)
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=

########################VERIFICANDO PROPORÇÃO DE INADIMPLENTES
prop.table(table(dados_clientes$inadimplente))

########################DIVIDINDO DATASET PARA TREINO E TESTE
#separando 75% dos dados para treino
indice <- createDataPartition(dados_clientes$inadimplente,p=0.75,list = FALSE)
dim(indice)
dados_treino <- dados_clientes[indice,]
table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))

#atribuindo os outros 25% dos dados para testes
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)
# -=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=
########################MODELO DE MACHINE LEARNING

#Construindo a primeira versão randomForest
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_teste)
modelo_v1

#avaliando modelo
plot(modelo_v1)

#previsoes com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1,dados_teste$inadimplente,positive = "1")
cm_v1

#Calculando precision, recall e F1-Score, métricas de avaliação de modelo preditivo
y <- dados_teste$inadimplente
y_pred_V1 <- previsoes_v1

precision <- posPredValue(y_pred_V1,y)
precision

recall <- sensitivity(y_pred_V1,y)
recall

F1 <- (2* precision * recall) / (precision + recall)
F1

#Importancia das variaveis para as previsões
varImpPlot(modelo_v1)

#Construindo a segunda versao apenas com as variaveis mais importantes
colnames(dados_treino)
modelo_v2 <- randomForest(inadimplente ~ PAY_0 + BILL_AMT1 + BILL_AMT2 + LIMIT_BAL + 
                            BILL_AMT5 + BILL_AMT3 + PAY_AMT1 + BILL_AMT4 + BILL_AMT6,
                          data = dados_treino)
modelo_v2

#avaliando modelo 2
plot(modelo_v2)

#previsoes com dados de teste versao 2
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix 2
cm_v2 <- caret::confusionMatrix(previsoes_v2,dados_teste$inadimplente,positive = "1")
cm_v2

#Calculando precision, recall e F1-Score, métricas de avaliação de modelo preditivo 2
y <- dados_teste$inadimplente
y_pred_V2 <- previsoes_v2

precision <- posPredValue(y_pred_V2,y)
precision

recall <- sensitivity(y_pred_V2,y)
recall

F1 <- (2* precision * recall) / (precision + recall)
F1

#Salvando modelos em disco
saveRDS(modelo_v1,file = "modelo/modelo_v1.rds")
saveRDS(modelo_v2,file = "modelo/modelo_v2.rds")

#Carregando modelo
modelo1 <- readRDS("modelo/modelo_v1.rds")
modelo2 <- readRDS("modelo/modelo_v2.rds")