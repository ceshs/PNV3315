#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib as plt


# In[2]:


## Import do arquivo, conversão em array
nome_arquivo = 'Cotas_2.xlsx'
## Import do arquivo, conversão em array e manipulação da tabela de cotas para transoformação em caixa
def le_arquivo(nome_arquivo):
    cotas = pd.read_excel(f"{os.getcwd()}\\{nome_arquivo}",).fillna(0).to_numpy()
    cotas=cotas[:-1,:]
    cotas=np.delete(cotas,1,1)
    return cotas

def Spline(tabela):
    # Armazena os valores intermediários de cada linha
    valores_intermediarios = []
    
    # Primeira linha da nova matriz
    nova_linha = [tabela[0,0], tabela[0,1]]
    for coluna in range(2, len(tabela[0])):
        valor_medio = tabela[0,coluna] - (tabela[0,coluna] - tabela[0,coluna-1]) / 2
        nova_linha.extend([valor_medio, tabela[0,coluna]])
    
    valores_intermediarios.append(nova_linha)
    
    # Processamento das linhas subsequentes
    for i in range(1, len(tabela)):
        # Calcula diferenças
        diferencas = []
        contador_inicial = len(tabela[0])
        for coluna in range(2, len(tabela[0])):
            h = tabela[0,coluna] - tabela[0,coluna-1]
            if tabela[i,coluna] - tabela[i,coluna-1] != 0:
                diferencas.append(h)
                contador_inicial -= 1
        
        # Matriz A e vetor b
        A = np.zeros((len(diferencas)-1, len(diferencas)+1))
        b = np.zeros(len(diferencas)-1)
        for m in range(len(diferencas)-1):
            A[m,m] = diferencas[m]
            A[m,m+1] = 2 * (diferencas[m] + diferencas[m+1])
            A[m,m+2] = diferencas[m+1]
            
            b[m] = (tabela[i,m+contador_inicial+1] - tabela[i,m+contador_inicial]) / diferencas[m+1] - \
                   (tabela[i,m+contador_inicial] - tabela[i,m+contador_inicial-1]) / diferencas[m]
        A = A[:,1:-1]  # Remove primeira e última coluna
        b *= 6
        
        # Resolvendo sistema linear para os coeficientes
        coeficientes = np.linalg.solve(A, b)
        coeficientes = np.append(coeficientes, 0.0)  # Adiciona zeros no início e fim
        coeficientes = np.insert(coeficientes, 0, 0.0, axis=0)
        
        # Calcula splines e preenche nova linha
        nova_linha = [tabela[i,0], tabela[i,1]]
        contador_zeros = 2
        for k in range(2, len(tabela[0])):
            if tabela[i,k] - tabela[i,k-1] == 0:
                nova_linha.extend([0.0, tabela[i,k]])
                contador_zeros += 1
            else:
                a = (coeficientes[k-contador_zeros+1] - coeficientes[k-contador_zeros]) / (6 * diferencas[k-contador_zeros])
                b = coeficientes[k-contador_zeros+1] / 2
                c = (tabela[i,k] - tabela[i,k-1]) / diferencas[k-contador_zeros] + \
                    (2 * diferencas[k-contador_zeros] * coeficientes[k-contador_zeros+1] + \
                     coeficientes[k-contador_zeros] * diferencas[k-contador_zeros]) / 6
                d = tabela[i,k]
                
                valor_medio = tabela[0,k] - (tabela[0,k] - tabela[0,k-1]) / 2
                s = a * (valor_medio - tabela[0,k])**3 + b * (valor_medio - tabela[0,k])**2 + \
                    c * (valor_medio - tabela[0,k]) + d

                nova_linha.extend([s, tabela[i,k]])
        
        valores_intermediarios.append(nova_linha)
    
    # Converte lista final para matriz
    tabela = np.asarray(valores_intermediarios)

    matriz_interpolada = []
    
    # Formando a primeira coluna da nova matriz
    primeira_coluna = [tabela[0, 0], tabela[1, 0]]
    for i in range(2, len(tabela)):
        valor_intermediario = tabela[i, 0] - (tabela[i, 0] - tabela[i - 1, 0]) / 2
        primeira_coluna.append(valor_intermediario)
        primeira_coluna.append(tabela[i, 0])
    
    matriz_interpolada.append(primeira_coluna)
    
    # Formando todas as novas colunas da nova matriz
    for j in range(1, len(tabela[0])):
        alturas = []
        indice_inicio = len(tabela)
        for i in range(2, len(tabela)):
            delta_h = tabela[i, 0] - tabela[i - 1, 0]
            if tabela[i, j] - tabela[i - 1, j] != 0:
                alturas.append(delta_h)
                indice_inicio -= 1
            elif tabela[i, j] - tabela[i - 1, j] == 0 and i > len(tabela) / 2:
                indice_inicio -= 1

        # Cálculo da matriz A e vetor b
        matriz_A = np.zeros((len(alturas) - 1, len(alturas) + 1))
        vetor_b = np.zeros(len(alturas) - 1)
        for m in range(len(alturas) - 1):
            matriz_A[m, m] = alturas[m]
            matriz_A[m, m + 1] = 2 * (alturas[m] + alturas[m + 1])
            matriz_A[m, m + 2] = alturas[m + 1]
            vetor_b[m] = (tabela[m + indice_inicio + 1, j] - tabela[m + indice_inicio, j]) / alturas[m + 1] - \
                         (tabela[m + indice_inicio, j] - tabela[m + indice_inicio - 1, j]) / alturas[m]
        matriz_A = matriz_A[:, 1:len(matriz_A[0]) - 1]  # Removendo primeira e última coluna para a operação
        vetor_b = 6 * vetor_b
        
        # Cálculo da matriz de coeficientes g
        coeficientes_g = np.linalg.solve(matriz_A, vetor_b)
        coeficientes_g = np.append(coeficientes_g, 0.0)  # Adicionando o primeiro e último zero que foram retirados
        coeficientes_g = np.insert(coeficientes_g, 0, 0.0, axis=0)
        
        # Cálculo das splines entre cada ponto e seu valor
        nova_coluna = [tabela[0, j], tabela[1, j]]
        contador_zeros = 2
        for k in range(2, len(tabela)):
            if tabela[k, j] - tabela[k - 1, j] == 0:
                nova_coluna.append(0.0)
                nova_coluna.append(tabela[k, j])
                contador_zeros += 1
            else:
                a = (coeficientes_g[k - contador_zeros + 1] - coeficientes_g[k - contador_zeros]) / (6 * alturas[k - contador_zeros])
                b = coeficientes_g[k - contador_zeros + 1] / 2
                c = (tabela[k, j] - tabela[k - 1, j]) / alturas[k - contador_zeros] + \
                    (2 * alturas[k - contador_zeros] * coeficientes_g[k - contador_zeros + 1] + coeficientes_g[k - contador_zeros] * alturas[k - contador_zeros]) / 6
                d = tabela[k, j]
                
                valor_intermediario = tabela[k, 0] - (tabela[k, 0] - tabela[k - 1, 0]) / 2
                s = a * (valor_intermediario - tabela[k, 0])**3 + b * (valor_intermediario - tabela[k, 0])**2 + \
                    c * (valor_intermediario - tabela[k, 0]) + d
                nova_coluna.append(s)
                nova_coluna.append(tabela[k, j])
        
        matriz_interpolada.append(nova_coluna)
    
    # Transformando a lista em matriz e transpondo para obter ela em forma de colunas
    tabela = np.asarray(matriz_interpolada).transpose()
    return tabela


def SplineMultipla(cotas, qnt):
    for _ in range(1,qnt+1):
        cotas=Spline(cotas)
    return cotas


#-------------Definição do Calado e seu índice na tabela de Cotas-----------------------#


def chooseCalado(tabelaCotas, calado):
    
    caladoIndex=0
    for j in range(len(tabelaCotas[0])):
        if tabelaCotas[0,j]==calado:
            caladoIndex=j
        elif tabelaCotas[0,j]<calado and tabelaCotas[0,j+1]>calado:
            caladoIndex=j
    
    caladoTab=tabelaCotas[0,caladoIndex]
    print("\nCalado Considerado:",caladoTab)
    return caladoIndex, caladoTab


# In[3]:


def chooseCalado(tabela_profundidade, profundidade):
    indice_profundidade = 0
    for i, p in enumerate(tabela_profundidade[0]):
        if p == profundidade:
            indice_profundidade = i
        elif tabela_profundidade[0, i] < profundidade < tabela_profundidade[0, i + 1]:
            indice_profundidade = i
    valor_profundidade = tabela_profundidade[0, indice_profundidade]
    print("\nProfundidade Considerada:", valor_profundidade)
    return indice_profundidade, valor_profundidade


# In[8]:


def calc_paineis_laterais(tabelaCotas,caladoIndex):
    #i representa as linhas
    #j representa as colunas
    storeVectorA=[]
    storeScalarA=[]
    storeC=[]
    #--------------------Cálculo dos Painéis Laterais---------------------------- #   
    for i in range(1,len(tabelaCotas)-1):
        for j in range(2,caladoIndex+1):
            
            #Vetores painéis
            v1=np.array([0,tabelaCotas[i,j]-tabelaCotas[i,j-1],tabelaCotas[0,j]-tabelaCotas[0,j-1]]) #p2-p1
            v2=np.array([tabelaCotas[i+1,0]-tabelaCotas[i,0],tabelaCotas[i+1,j-1]-tabelaCotas[i+1,j],0]) #p4-p1
            v3=np.array([0,tabelaCotas[i+1,j-1]-tabelaCotas[i+1,j],tabelaCotas[0,j-1]-tabelaCotas[0,j]]) #p4-p3
            v4=np.array([tabelaCotas[i,0]-tabelaCotas[i+1,0],tabelaCotas[i,j]-tabelaCotas[i+1,j],0]) #p2-p3
            
            #Calculo vetor A e C do painel sendo analisado
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i+1,0])/4,(tabelaCotas[i,j]+tabelaCotas[i,j-1]+tabelaCotas[i+1,j-1]+tabelaCotas[i+1,j])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j-1])/4])
            
            #Guardando A em forma vetor e escalar e C vetor em listas
            storeVectorA.append(currentA.copy())
            storeScalarA.append(np.linalg.norm(currentA))
            storeC.append(currentC)
            
            #Para o outro lado
            v1=np.array([tabelaCotas[i+1,0]-tabelaCotas[i,0],(-tabelaCotas[i+1,j-1])-(-tabelaCotas[i,j-1]),0]) #p2-p1
            v2=np.array([0,(-tabelaCotas[i,j])-(-tabelaCotas[i,j-1]),tabelaCotas[0,j]-tabelaCotas[0,j-1]]) #p4-p1
            v3=np.array([tabelaCotas[i,0]-tabelaCotas[i+1,0],(-tabelaCotas[i,j])-(-tabelaCotas[i+1,j]),0]) #p4-p3
            v4=np.array([0,(-tabelaCotas[i+1,j-1])-(-tabelaCotas[i+1,j]),tabelaCotas[0,j-1]-tabelaCotas[0,j]]) #p2-p3
            
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i+1,0])/4,(-tabelaCotas[i,j]-tabelaCotas[i,j-1]-tabelaCotas[i+1,j-1]-tabelaCotas[i+1,j])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j-1])/4])
            
            storeVectorA.append(currentA.copy())
            storeScalarA.append(np.linalg.norm(currentA))
            storeC.append(currentC.copy())
    return storeVectorA, storeScalarA, storeC


def calc_paineis_popa(tabelaCotas,caladoIndex):
    storeVectorPopA=[]
    storeScalarPopA=[]
    storePopC=[]
    
    for j in range(1,caladoIndex):
        v1=np.array([0,0,tabelaCotas[0,j+1]-tabelaCotas[0,j]])#p2-p1
        v2=np.array([0,tabelaCotas[1,j],0])#p4-p1
        v3=np.array([0,tabelaCotas[1,j]-tabelaCotas[1,j+1],tabelaCotas[0,j]-tabelaCotas[0,j+1]])#p4-p3
        v4=np.array([0,0-tabelaCotas[1,j+1],0])#p2-p3
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([0,(tabelaCotas[1,j]+tabelaCotas[1,j+1])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j+1])/4])
        
        storeVectorPopA.append(currentA.copy())
        storeScalarPopA.append(np.linalg.norm(currentA))
        storePopC.append(currentC.copy())
        
        #Do outro lado
        v1=np.array([0,-tabelaCotas[1,j],0])#p2-p1
        v2=np.array([0,0,tabelaCotas[0,j+1]-tabelaCotas[0,j]])#p4-p1
        v3=np.array([0,tabelaCotas[1,j+1],0])#p4-p3
        v4=np.array([0,-tabelaCotas[1,j]-(-tabelaCotas[1,j+1]),tabelaCotas[0,j]-tabelaCotas[0,j+1]])#p2-p3
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([0,(-tabelaCotas[1,j]-tabelaCotas[1,j+1])/4,(2*tabelaCotas[0,j]+2*tabelaCotas[0,j+1])/4])
        
        storeVectorPopA.append(currentA.copy())
        storeScalarPopA.append(np.linalg.norm(currentA))
        storePopC.append(currentC.copy())
    return storeVectorPopA, storeScalarPopA, storePopC

def calc_paineis_topo(tabelaCotas, caladoIndex, caladoTab):
    storeVectorTopA=[]
    storeScalarTopA=[]
    storeTopC=[]
    storeBal=[]
    
    #Formando a matriz do plano da linha d'água matrixYWL
    matrixYWL=tabelaCotas[1:,:caladoIndex+1] #É extraído a primeira coluna com os valores de x até a coluna com a spline da linha d'água do calado
    matrixYWL=np.delete(matrixYWL,np.s_[1:len(matrixYWL[0])-1],axis=1) #Deletamos todas as colunas entre essas duas colunas
    
    #Loop para obter valores igualmente espaçados do y=0 até y=spline para cada baliza
    for i in range(len(matrixYWL)):
        currentBal=np.linspace(0,matrixYWL[i,-1],len(tabelaCotas[0])*5) #Aqui é definido a malha do plano de flutuação
        currentBal=np.insert(currentBal,0,matrixYWL[i,0],axis=0)
        storeBal.append(currentBal)
    
    #Transformando lista em matriz    
    matrixYWL=np.asarray(storeBal)
    
    #Cálculo do vetores e painéis
    for i in range(len(matrixYWL)-1):
        for j in range(2,len(matrixYWL[0])):
            v1=np.array([matrixYWL[i+1,0]-matrixYWL[i,0],matrixYWL[i+1,j-1]-matrixYWL[i,j-1],caladoTab])#p2-p1
            v2=np.array([0,matrixYWL[i,j]-matrixYWL[i,j-1],caladoTab])#p4-p1
            v3=np.array([matrixYWL[i,0]-matrixYWL[i+1,0],matrixYWL[i,j]-matrixYWL[i+1,j],caladoTab])#p4-p3
            v4=np.array([0,matrixYWL[i+1,j-1]-matrixYWL[i+1,j],caladoTab])#p2-p3
            
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*matrixYWL[i,0]+2*matrixYWL[i+1,0])/4,(matrixYWL[i,j]+matrixYWL[i,j-1]+matrixYWL[i+1,j-1]+matrixYWL[i+1,j])/4,caladoTab])
            
            storeVectorTopA.append(currentA.copy())
            storeScalarTopA.append(np.linalg.norm(currentA))
            storeTopC.append(currentC.copy())
    
            #Para o outro lado
            v1=np.array([matrixYWL[i+1,0]-matrixYWL[i,0],-matrixYWL[i+1,j]-(-matrixYWL[i,j]),caladoTab])#p2-p1
            v2=np.array([0,-matrixYWL[i,j-1]-(-matrixYWL[i,j]),caladoTab])#p4-p1
            v3=np.array([matrixYWL[i,0]-matrixYWL[i+1,0],-matrixYWL[i,j-1]-(-matrixYWL[i+1,j-1]),caladoTab])#p4-p3
            v4=np.array([0,-matrixYWL[i+1,j]-(-matrixYWL[i+1,j-1]),caladoTab])#p2-p3
            
            currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            currentC=np.array([(2*matrixYWL[i,0]+2*matrixYWL[i+1,0])/4,(-matrixYWL[i,j]-matrixYWL[i,j-1]-matrixYWL[i+1,j-1]-matrixYWL[i+1,j])/4,caladoTab])
                     
            storeVectorTopA.append(currentA.copy())
            storeScalarTopA.append(np.linalg.norm(currentA))
            storeTopC.append(currentC.copy())
            
    return storeVectorTopA,storeScalarTopA,storeTopC

def calc_paineis_fundo(tabelaCotas):
    storeVectorBotA=[]
    storeScalarBotA=[]
    storeBotC=[]
    
    for i in range(2,len(tabelaCotas)):
        v1=np.array([tabelaCotas[i,0]-tabelaCotas[i-1,0],tabelaCotas[i,1]-tabelaCotas[i-1,1],0])
        v2=np.array([0,-tabelaCotas[i-1,1],0])
        v3=np.array([tabelaCotas[i-1,0]-tabelaCotas[i,0],0,0])#p4-p3
        v4=np.array([0,tabelaCotas[i,1],0])
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i-1,0])/4,(tabelaCotas[i,1]+tabelaCotas[i-1,1])/4,(2*tabelaCotas[0,1])/4])
        
        storeVectorBotA.append(currentA.copy())
        storeScalarBotA.append(np.linalg.norm(currentA))
        storeBotC.append(currentC.copy())
        
        v1=np.array([tabelaCotas[i,0]-tabelaCotas[i-1,0],0,0])#p2-p1
        v2=np.array([0,tabelaCotas[i-1,1],0])#p4-p1
        v3=np.array([tabelaCotas[i-1,0]-tabelaCotas[i,0],-tabelaCotas[i-1,1]-(-tabelaCotas[i,1]),0])#p4-p3
        v4=np.array([0,0-(-tabelaCotas[i,1]),0])#p2-p3
        
        currentA=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        currentC=np.array([(2*tabelaCotas[i,0]+2*tabelaCotas[i-1,0])/4,(-tabelaCotas[i,1]-tabelaCotas[i-1,1])/4,(2*tabelaCotas[0,1])/4])
        
        storeVectorBotA.append(currentA.copy())
        storeScalarBotA.append(np.linalg.norm(currentA))
        storeBotC.append(currentC.copy())
    return storeVectorBotA,storeScalarBotA,storeBotC

def calcula_sw(storeScalarA,storeScalarBotA,storeScalarPopA):
    wetS=0
    wetS+=np.sum(storeScalarA)
    wetS+=np.sum(storeScalarBotA)
    wetS+=np.sum(storeScalarPopA)
    
    return wetS    

def calcula_aw(storeVectorTopA):
    wetA=0
    storeVectorTopAarray=np.asarray(storeVectorTopA)
    wetA=storeVectorTopAarray[:,2].sum(axis=0)
    
    return wetA

def calcula_nabla_desloc(storeVectorA,storeBotC,storeVectorPopA,storeVectorBotA,storeVectorTopA,storeTopC,storeC,storePopC):
    xTerm, yTerm, zTerm = 0, 0, 0
    
    for i in range(len(storeVectorA)):
        xTerm+=storeVectorA[i][0]*storeC[i][0]
        yTerm+=storeVectorA[i][1]*storeC[i][1]
        zTerm+=storeVectorA[i][2]*storeC[i][2]    
    
    for i in range(len(storeVectorPopA)):
        xTerm+=storeVectorPopA[i][0]*storePopC[i][0]
        yTerm+=storeVectorPopA[i][1]*storePopC[i][1]
        zTerm+=storeVectorPopA[i][2]*storePopC[i][2]
       
    for i in range(len(storeVectorBotA)):
        xTerm+=storeVectorBotA[i][0]*storeBotC[i][0]
        yTerm+=storeVectorBotA[i][1]*storeBotC[i][1]
        zTerm+=storeVectorBotA[i][2]*storeBotC[i][2]
    
    for i in range(len(storeVectorTopA)):
        xTerm+=storeVectorTopA[i][0]*storeTopC[i][0]
        yTerm+=storeVectorTopA[i][1]*storeTopC[i][1]
        zTerm+=storeVectorTopA[i][2]*storeTopC[i][2]
         
    nabla=(xTerm+yTerm+zTerm)/3
    desloc=nabla*1.025
    return nabla,desloc

def calcula_lcf_tcf(storeVectorTopA,storeTopC):
    LCFnumerador, LCFandTCFdenominador,TCFnumerador = 0, 0, 0
    
    for i in range(len(storeVectorTopA)):
        
        LCFnumerador+=(-storeVectorTopA[i][2])*storeTopC[i][0]
        TCFnumerador+=(-storeVectorTopA[i][2])*storeTopC[i][1]   
    
        LCFandTCFdenominador+=(-storeVectorTopA[i][2])
    
    LCF=LCFnumerador/LCFandTCFdenominador
    TCF=TCFnumerador/LCFandTCFdenominador
    
    return LCF,TCF

def calcula_inercia(storeVectorTopA,storeTopC,TCF,LCF):
    inertiaL, inertiaT = 0, 0
    
    for i in range(len(storeVectorTopA)):
        #Lembrar delft usa a notação trocada
        inertiaT+=(storeVectorTopA[i][2]*(storeTopC[i][1]-TCF)**2)
        inertiaL+=(storeVectorTopA[i][2]*(storeTopC[i][0]-LCF)**2)
    
    return inertiaT,inertiaL
    
def calc_bm(nabla,storeVectorA,storeC,storeVectorPopA,storePopC,storeVectorBotA,storeBotC,storeVectorTopA,storeTopC,inertiaL,inertiaT):
    LCB, TCB, KB = 0, 0, 0
    
    for i in range(len(storeVectorA)):
        LCB+=(storeVectorA[i][0]*storeC[i][0]*storeC[i][0]/2)
        TCB+=(storeVectorA[i][1]*storeC[i][1]*storeC[i][1]/2)
        KB+=(storeVectorA[i][2]*storeC[i][2]*storeC[i][2]/2)
    
    for i in range(len(storeVectorPopA)):
        LCB+=(storeVectorPopA[i][0]*storePopC[i][0]*storePopC[i][0]/2)
        TCB+=(storeVectorPopA[i][1]*storePopC[i][1]*storePopC[i][1]/2)
        KB+=(storeVectorPopA[i][2]*storePopC[i][2]*storePopC[i][2]/2)
    
    for i in range(len(storeVectorBotA)):
        LCB+=(storeVectorBotA[i][0]*storeBotC[i][0]*storeBotC[i][0]/2)
        TCB+=(storeVectorBotA[i][1]*storeBotC[i][1]*storeBotC[i][1]/2)
        KB+=(storeVectorBotA[i][2]*storeBotC[i][2]*storeBotC[i][2]/2)
        
    for i in range(len(storeVectorTopA)):
        LCB+=(storeVectorTopA[i][0]*storeTopC[i][0]*storeTopC[i][0]/2)
        TCB+=(storeVectorTopA[i][1]*storeTopC[i][1]*storeTopC[i][1]/2)
        KB+=(storeVectorTopA[i][2]*storeTopC[i][2]*storeTopC[i][2]/2)
       
    LCB=LCB/nabla
    TCB=TCB/nabla
    KB=KB/nabla
    BML=inertiaL/nabla
    BMT=inertiaT/nabla
    return LCB,TCB,KB,BML,BMT

def wrapper_infos(tabelaCotas, caladoIndex, caladoTab):

    storeVectorA, storeScalarA, storeC = calc_paineis_laterais(tabelaCotas,caladoIndex)
    storeVectorPopA, storeScalarPopA, storePopC = calc_paineis_popa(tabelaCotas,caladoIndex)
    storeVectorTopA,storeScalarTopA,storeTopC = calc_paineis_topo(tabelaCotas, caladoIndex, caladoTab)
    storeVectorBotA,storeScalarBotA,storeBotC = calc_paineis_fundo(tabelaCotas)
    wetS = calcula_sw(storeScalarA,storeScalarBotA,storeScalarPopA)
    wetA = calcula_aw(storeVectorTopA)
    nabla, desloc = calcula_nabla_desloc(storeVectorA,storeBotC,storeVectorPopA,storeVectorBotA,storeVectorTopA,storeTopC,storeC,storePopC)
    lcf, tcf = calcula_lcf_tcf(storeVectorTopA,storeTopC)
    inercia_transversal, inercia_longitunidal = calcula_inercia(storeVectorTopA,storeTopC,tcf,lcf)
    LCB,TCB,KB,BML,BMT = calc_bm(nabla,storeVectorA,storeC,storeVectorPopA,storePopC,storeVectorBotA,storeBotC,storeVectorTopA,storeTopC,inercia_transversal,inercia_longitunidal)
    return (wetS , wetA, nabla, lcf, tcf, inercia_transversal, inercia_longitunidal, LCB, TCB, KB, BML, BMT, desloc)


# In[ ]:


#------------------------------------------------------------------------------------#    

#Aproximando a nossa tabela de Cotas
def retorna_calculos_ep(nome_arquivo)
    tabelaCotas=le_arquivo(nome_arquivo)
    #Fazendo quatro interpolções para cada direação. Após este número de interpolações não há mudanças significativas nos valores hidroestáticos
    tabelaCotas=SplineMultipla(tabelaCotas, 4)
    resultsPerCalado=[]
    inputs=np.arange(0.75,3.25,0.25)
    list_calados = []
    for i in range(len(inputs)):
        resultsCalado=chooseCalado(tabelaCotas, inputs[i])
        caladoIndex, caladoTab = resultsCalado[0], resultsCalado[1]
        #hidroProps
        resultsPerCalado.append(wrapper_infos(tabelaCotas, caladoIndex, caladoTab))
        list_calados.append(resultsCalado[1])
    data=[]
    inputs=inputs.tolist()
    parameters=["Área da superfície molhada (Sw) [m2]","Área do plano de linha d'água (Awl) [m2]","Volume Deslocado [m3]","LCF [m]","TCF [m]","Inércia transversal (IT) [m4]",
                "Inércia longitudinal (IL) [m4]","LCB [m]","TCB [m]","KB [m]","BML [m]","BMT [m]","Deslocamento (Δ) [t]"]

    #Loop para obter cada propriedades separadas em listas

    retorna_planilha_dados = pd.DataFrame(resultsPerCalado)
    retorna_planilha_dados.columns = parameters
    retorna_planilha_dados  = retorna_planilha_dados.T
    retorna_planilha_dados.columns = list_calados
    retorna_planilha_dados.to_excel('Table_cotas.xlsx') 
    return True
#Loop para plotar todos os gráficos
# for i in range(len(parameters)):
#     plt.plot(inputs,data[i])
#     plt.xlabel("Calado")
#     plt.ylabel(str(parameters[i]))
#     plt.title(str(parameters[i]) + "  X Calado [m]")
# plt.show()
  
# #Loop para printar as listas com o valores de cada propriedade por calado      
# print("\nCalados:", inputs)
# for i in range(len(data)):
#     print("\n"+str(parameters[i])+":",data[i])


# In[9]:





# In[10]:





# In[34]:





# In[33]:


resultsCalado[1]


# In[17]:


list_final


# In[ ]:




