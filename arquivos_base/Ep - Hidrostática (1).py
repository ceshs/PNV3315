#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib as plt


# In[8]:


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




def picker(tabela_profundidade, profundidade):
    indice_profundidade = 0
    for i, p in enumerate(tabela_profundidade[0]):
        if p == profundidade:
            indice_profundidade = i
        elif tabela_profundidade[0, i] < profundidade < tabela_profundidade[0, i + 1]:
            indice_profundidade = i
    
    valor_profundidade = tabela_profundidade[0, indice_profundidade]
    return indice_profundidade, valor_profundidade

def calcula_paineis_laterais(tabela_cotas, indice_calado):
    vetores_armazenados = []
    escalares_armazenados = []
    Cs_armazenados = []
    
    #--------------------Cálculo dos Painéis Laterais---------------------------- #   
    for i in range(1, len(tabela_cotas) - 1):
        for j in range(2, indice_calado + 1):
            
            # Vetores painéis
            v1 = np.array([0, tabela_cotas[i, j] - tabela_cotas[i, j - 1], tabela_cotas[0, j] - tabela_cotas[0, j - 1]]) # p2-p1
            v2 = np.array([tabela_cotas[i + 1, 0] - tabela_cotas[i, 0], tabela_cotas[i + 1, j - 1] - tabela_cotas[i + 1, j], 0]) # p4-p1
            v3 = np.array([0, tabela_cotas[i + 1, j - 1] - tabela_cotas[i + 1, j], tabela_cotas[0, j - 1] - tabela_cotas[0, j]]) # p4-p3
            v4 = np.array([tabela_cotas[i, 0] - tabela_cotas[i + 1, 0], tabela_cotas[i, j] - tabela_cotas[i + 1, j], 0]) # p2-p3
            
            # Cálculo vetor A e C do painel sendo analisado
            atual_A = 0.5 * (np.cross(v1, v2) + np.cross(v3, v4))
            atual_C = np.array([(2 * tabela_cotas[i, 0] + 2 * tabela_cotas[i + 1, 0]) / 4, (tabela_cotas[i, j] + tabela_cotas[i, j - 1] + tabela_cotas[i + 1, j - 1] + tabela_cotas[i + 1, j]) / 4, (2 * tabela_cotas[0, j] + 2 * tabela_cotas[0, j - 1]) / 4])
            
            # Guardando A em forma vetor e escalar e C vetor em listas
            vetores_armazenados.append(atual_A.copy())
            escalares_armazenados.append(np.linalg.norm(atual_A))
            Cs_armazenados.append(atual_C)
            
            # Para o outro lado
            v1 = np.array([tabela_cotas[i + 1, 0] - tabela_cotas[i, 0], (-tabela_cotas[i + 1, j - 1]) - (-tabela_cotas[i, j - 1]), 0]) # p2-p1
            v2 = np.array([0, (-tabela_cotas[i, j]) - (-tabela_cotas[i, j - 1]), tabela_cotas[0, j] - tabela_cotas[0, j - 1]]) # p4-p1
            v3 = np.array([tabela_cotas[i, 0] - tabela_cotas[i + 1, 0], (-tabela_cotas[i, j]) - (-tabela_cotas[i + 1, j]), 0]) # p4-p3
            v4 = np.array([0, (-tabela_cotas[i + 1, j - 1]) - (-tabela_cotas[i + 1, j]), tabela_cotas[0, j - 1] - tabela_cotas[0, j]]) # p2-p3
            
            atual_A = 0.5 * (np.cross(v1, v2) + np.cross(v3, v4))
            atual_C = np.array([(2 * tabela_cotas[i, 0] + 2 * tabela_cotas[i + 1, 0]) / 4, (-tabela_cotas[i, j] - tabela_cotas[i, j - 1] - tabela_cotas[i + 1, j - 1] - tabela_cotas[i + 1, j]) / 4, (2 * tabela_cotas[0, j] + 2 * tabela_cotas[0, j - 1]) / 4])
            
            vetores_armazenados.append(atual_A.copy())
            escalares_armazenados.append(np.linalg.norm(atual_A))
            Cs_armazenados.append(atual_C.copy())
    return vetores_armazenados, escalares_armazenados, Cs_armazenados


def calcula_paineis_popa(tabela_cotas, indice_calado):
    armazenar_vetor_popa = []
    armazenar_escalar_popa = []
    armazenar_popa_c = []
    
    for j in range(1, indice_calado):
        v1 = np.array([0, 0, tabela_cotas[0, j+1] - tabela_cotas[0, j]])  # p2-p1
        v2 = np.array([0, tabela_cotas[1, j], 0])  # p4-p1
        v3 = np.array([0, tabela_cotas[1, j] - tabela_cotas[1, j+1], tabela_cotas[0, j] - tabela_cotas[0, j+1]])  # p4-p3
        v4 = np.array([0, 0 - tabela_cotas[1, j+1], 0])  # p2-p3
        
        corrente_a = 0.5 * (np.cross(v1, v2) + np.cross(v3, v4))
        corrente_c = np.array([0, (tabela_cotas[1, j] + tabela_cotas[1, j+1]) / 4, (2 * tabela_cotas[0, j] + 2 * tabela_cotas[0, j+1]) / 4])
        
        armazenar_vetor_popa.append(corrente_a.copy())
        armazenar_escalar_popa.append(np.linalg.norm(corrente_a))
        armazenar_popa_c.append(corrente_c.copy())
        
        # Do outro lado
        v1 = np.array([0, -tabela_cotas[1, j], 0])  # p2-p1
        v2 = np.array([0, 0, tabela_cotas[0, j+1] - tabela_cotas[0, j]])  # p4-p1
        v3 = np.array([0, tabela_cotas[1, j+1], 0])  # p4-p3
        v4 = np.array([0, -tabela_cotas[1, j] - (-tabela_cotas[1, j+1]), tabela_cotas[0, j] - tabela_cotas[0, j+1]])  # p2-p3
        
        corrente_a = 0.5 * (np.cross(v1, v2) + np.cross(v3, v4))
        corrente_c = np.array([0, (-tabela_cotas[1, j] - tabela_cotas[1, j+1]) / 4, (2 * tabela_cotas[0, j] + 2 * tabela_cotas[0, j+1]) / 4])
        
        armazenar_vetor_popa.append(corrente_a.copy())
        armazenar_escalar_popa.append(np.linalg.norm(corrente_a))
        armazenar_popa_c.append(corrente_c.copy())
    return armazenar_vetor_popa, armazenar_escalar_popa, armazenar_popa_c

def calcula_paineis_topo(tabela_cotas, indice_calado, calado_tab):
    store_vetor_topo_a=[]
    store_escalar_topo_a=[]
    store_topo_c=[]
    store_bal=[]
    
    #Formando a matriz do plano da linha d'água matrixYWL
    matrix_ywl=tabela_cotas[1:,:indice_calado+1] #É extraído a primeira coluna com os valores de x até a coluna com a spline da linha d'água do calado
    matrix_ywl=np.delete(matrix_ywl,np.s_[1:len(matrix_ywl[0])-1],axis=1) #Deletamos todas as colunas entre essas duas colunas
    
    #Loop para obter valores igualmente espaçados do y=0 até y=spline para cada baliza
    for i in range(len(matrix_ywl)):
        current_bal=np.linspace(0,matrix_ywl[i,-1],len(tabela_cotas[0])*5) #Aqui é definido a malha do plano de flutuação
        current_bal=np.insert(current_bal,0,matrix_ywl[i,0],axis=0)
        store_bal.append(current_bal)
    
    #Transformando lista em matriz    
    matrix_ywl=np.asarray(store_bal)
    
    #Cálculo do vetores e painéis
    for i in range(len(matrix_ywl)-1):
        for j in range(2,len(matrix_ywl[0])):
            v1=np.array([matrix_ywl[i+1,0]-matrix_ywl[i,0],matrix_ywl[i+1,j-1]-matrix_ywl[i,j-1],calado_tab])#p2-p1
            v2=np.array([0,matrix_ywl[i,j]-matrix_ywl[i,j-1],calado_tab])#p4-p1
            v3=np.array([matrix_ywl[i,0]-matrix_ywl[i+1,0],matrix_ywl[i,j]-matrix_ywl[i+1,j],calado_tab])#p4-p3
            v4=np.array([0,matrix_ywl[i+1,j-1]-matrix_ywl[i+1,j],calado_tab])#p2-p3
            
            current_a=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            current_c=np.array([(2*matrix_ywl[i,0]+2*matrix_ywl[i+1,0])/4,(matrix_ywl[i,j]+matrix_ywl[i,j-1]+matrix_ywl[i+1,j-1]+matrix_ywl[i+1,j])/4,calado_tab])
            
            store_vetor_topo_a.append(current_a.copy())
            store_escalar_topo_a.append(np.linalg.norm(current_a))
            store_topo_c.append(current_c.copy())
    
            #Para o outro lado
            v1=np.array([matrix_ywl[i+1,0]-matrix_ywl[i,0],-matrix_ywl[i+1,j]-(-matrix_ywl[i,j]),calado_tab])#p2-p1
            v2=np.array([0,-matrix_ywl[i,j-1]-(-matrix_ywl[i,j]),calado_tab])#p4-p1
            v3=np.array([matrix_ywl[i,0]-matrix_ywl[i+1,0],-matrix_ywl[i,j-1]-(-matrix_ywl[i+1,j-1]),calado_tab])#p4-p3
            v4=np.array([0,-matrix_ywl[i+1,j]-(-matrix_ywl[i+1,j-1]),calado_tab])#p2-p3
            
            current_a=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
            current_c=np.array([(2*matrix_ywl[i,0]+2*matrix_ywl[i+1,0])/4,(-matrix_ywl[i,j]-matrix_ywl[i,j-1]-matrix_ywl[i+1,j-1]-matrix_ywl[i+1,j])/4,calado_tab])
                     
            store_vetor_topo_a.append(current_a.copy())
            store_escalar_topo_a.append(np.linalg.norm(current_a))
            store_topo_c.append(current_c.copy())
            
    return store_vetor_topo_a,store_escalar_topo_a,store_topo_c

def calcula_paineis_fundo(tabela_cotas):
    store_vetor_fundo_a=[]
    store_escalar_fundo_a=[]
    store_fundo_c=[]
    
    for i in range(2,len(tabela_cotas)):
        v1=np.array([tabela_cotas[i,0]-tabela_cotas[i-1,0],tabela_cotas[i,1]-tabela_cotas[i-1,1],0])
        v2=np.array([0,-tabela_cotas[i-1,1],0])
        v3=np.array([tabela_cotas[i-1,0]-tabela_cotas[i,0],0,0])#p4-p3
        v4=np.array([0,tabela_cotas[i,1],0])
        
        current_a=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        current_c=np.array([(2*tabela_cotas[i,0]+2*tabela_cotas[i-1,0])/4,(tabela_cotas[i,1]+tabela_cotas[i-1,1])/4,(2*tabela_cotas[0,1])/4])
        
        store_vetor_fundo_a.append(current_a.copy())
        store_escalar_fundo_a.append(np.linalg.norm(current_a))
        store_fundo_c.append(current_c.copy())
        
        v1=np.array([tabela_cotas[i,0]-tabela_cotas[i-1,0],0,0])#p2-p1
        v2=np.array([0,tabela_cotas[i-1,1],0])#p4-p1
        v3=np.array([tabela_cotas[i-1,0]-tabela_cotas[i,0],-tabela_cotas[i-1,1]-(-tabela_cotas[i,1]),0])#p4-p3
        v4=np.array([0,0-(-tabela_cotas[i,1]),0])#p2-p3
        
        current_a=0.5*(np.cross(v1,v2)+np.cross(v3,v4))
        current_c=np.array([(2*tabela_cotas[i,0]+2*tabela_cotas[i-1,0])/4,(-tabela_cotas[i,1]-tabela_cotas[i-1,1])/4,(2*tabela_cotas[0,1])/4])
        
        store_vetor_fundo_a.append(current_a.copy())
        store_escalar_fundo_a.append(np.linalg.norm(current_a))
        store_fundo_c.append(current_c.copy())
    return store_vetor_fundo_a,store_escalar_fundo_a,store_fundo_c

def calcula_superficie_molhada(store_escalar_a,store_escalar_fundo_a,store_escalar_pop_a):
    superficie_molhada=0
    superficie_molhada+=np.sum(store_escalar_a)
    superficie_molhada+=np.sum(store_escalar_fundo_a)
    superficie_molhada+=np.sum(store_escalar_pop_a)
    
    return superficie_molhada    

def calcula_area_molhada(store_vetor_top_a):
    area_molhada=0
    store_vetor_top_aarray=np.asarray(store_vetor_top_a)
    area_molhada=store_vetor_top_aarray[:,2].sum(axis=0)
    
    return area_molhada

def calcula_nabla_deslocamento(store_vetor_a,store_fundo_c,store_vetor_pop_a,store_vetor_fundo_a,store_vetor_top_a,store_top_c,store_c,store_pop_c):
    x_term, y_term, z_term = 0, 0, 0
    
    for i in range(len(store_vetor_a)):
        x_term+=store_vetor_a[i][0]*store_c[i][0]
        y_term+=store_vetor_a[i][1]*store_c[i][1]
        z_term+=store_vetor_a[i][2]*store_c[i][2]    
    
    for i in range(len(store_vetor_pop_a)):
        x_term+=store_vetor_pop_a[i][0]*store_pop_c[i][0]
        y_term+=store_vetor_pop_a[i][1]*store_pop_c[i][1]
        z_term+=store_vetor_pop_a[i][2]*store_pop_c[i][2]
       
    for i in range(len(store_vetor_fundo_a)):
        x_term+=store_vetor_fundo_a[i][0]*store_fundo_c[i][0]
        y_term+=store_vetor_fundo_a[i][1]*store_fundo_c[i][1]
        z_term+=store_vetor_fundo_a[i][2]*store_fundo_c[i][2]
    
    for i in range(len(store_vetor_top_a)):
        x_term+=store_vetor_top_a[i][0]*store_top_c[i][0]
        y_term+=store_vetor_top_a[i][1]*store_top_c[i][1]
        z_term+=store_vetor_top_a[i][2]*store_top_c[i][2]
         
    nabla=(x_term+y_term+z_term)/3
    deslocamento=nabla*1.025
    return nabla,deslocamento

def calcula_lcf_tcf(store_vetor_top_a,store_top_c):
    lcf_numerador, lcf_tcf_denominador, tcf_numerador = 0, 0, 0
    
    for i in range(len(store_vetor_top_a)):
        
        lcf_numerador+=(-store_vetor_top_a[i][2])*store_top_c[i][0]
        tcf_numerador+=(-store_vetor_top_a[i][2])*store_top_c[i][1]   
    
        lcf_tcf_denominador+=(-store_vetor_top_a[i][2])
    
    lcf=lcf_numerador/lcf_tcf_denominador
    tcf=tcf_numerador/lcf_tcf_denominador
    
    return lcf,tcf

def calcula_inercia(store_vetor_top_a,store_top_c,tcf,lcf):
    inercia_longitunidal, inercia_transversal = 0, 0
    
    for i in range(len(store_vetor_top_a)):
        #Lembrar delft usa a notação trocada
        inercia_transversal+=(store_vetor_top_a[i][2]*(store_top_c[i][1]-tcf)**2)
        inercia_longitunidal+=(store_vetor_top_a[i][2]*(store_top_c[i][0]-lcf)**2)
    
    return inercia_transversal,inercia_longitunidal
    
def calcula_bm(nabla,store_vetor_a,store_c,store_vetor_pop_a,store_pop_c,store_vetor_fundo_a,store_fundo_c,store_vetor_top_a,store_top_c,inercia_longitunidal,inercia_transversal,lcf, tcf):
    lcb, tcb, kb = 0, 0, 0
    
    for i in range(len(store_vetor_a)):
        lcb+=(store_vetor_a[i][0]*store_c[i][0]*store_c[i][0]/2)
        tcb+=(store_vetor_a[i][1]*store_c[i][1]*store_c[i][1]/2)
        kb+=(store_vetor_a[i][2]*store_c[i][2]*store_c[i][2]/2)
    
    for i in range(len(store_vetor_pop_a)):
        lcb+=(store_vetor_pop_a[i][0]*store_pop_c[i][0]*store_pop_c[i][0]/2)
        tcb+=(store_vetor_pop_a[i][1]*store_pop_c[i][1]*store_pop_c[i][1]/2)
        kb+=(store_vetor_pop_a[i][2]*store_pop_c[i][2]*store_pop_c[i][2]/2)
    
    for i in range(len(store_vetor_fundo_a)):
        lcb+=(store_vetor_fundo_a[i][0]*store_fundo_c[i][0]*store_fundo_c[i][0]/2)
        tcb+=(store_vetor_fundo_a[i][1]*store_fundo_c[i][1]*store_fundo_c[i][1]/2)
        kb+=(store_vetor_fundo_a[i][2]*store_fundo_c[i][2]*store_fundo_c[i][2]/2)
        
    for i in range(len(store_vetor_top_a)):
        lcb+=(store_vetor_top_a[i][0]*store_top_c[i][0]*store_top_c[i][0]/2)
        tcb+=(store_vetor_top_a[i][1]*store_top_c[i][1]*store_top_c[i][1]/2)
        kb+=(store_vetor_top_a[i][2]*store_top_c[i][2]*store_top_c[i][2]/2)
       
    lcb=lcb/nabla
    tcb=tcb/nabla
    kb=kb/nabla
    
    lcg=lcb-lcf
    tcg=tcb-tcf
    
    lcg+=inercia_transversal/nabla
    tcg+=inercia_longitunidal/nabla
    
    return lcb, tcb, kb, lcg, tcg

def wrapper_infos(tabelaCotas, caladoIndex, caladoTab):

    storeVectorA, storeScalarA, storeC = calcula_paineis_laterais(tabelaCotas,caladoIndex)
    storeVectorPopA, storeScalarPopA, storePopC = calcula_paineis_popa(tabelaCotas,caladoIndex)
    storeVectorTopA,storeScalarTopA,storeTopC = calcula_paineis_topo(tabelaCotas, caladoIndex, caladoTab)
    storeVectorBotA,storeScalarBotA,storeBotC = calcula_paineis_fundo(tabelaCotas)
    wetS = calcula_superficie_molhada(storeScalarA,storeScalarBotA,storeScalarPopA)
    wetA = calcula_area_molhada(storeVectorTopA)
    nabla, desloc = calcula_nabla_deslocamento(storeVectorA,storeBotC,storeVectorPopA,storeVectorBotA,storeVectorTopA,storeTopC,storeC,storePopC)
    lcf, tcf = calcula_lcf_tcf(storeVectorTopA,storeTopC)
    inercia_transversal, inercia_longitunidal = calcula_inercia(storeVectorTopA,storeTopC,tcf,lcf)
    LCB,TCB,KB,BML,BMT = calcula_bm(nabla,storeVectorA,storeC,storeVectorPopA,storePopC,storeVectorBotA,storeBotC,storeVectorTopA,storeTopC,inercia_transversal,inercia_longitunidal,lcf, tcf)
    return (wetS , wetA, nabla, lcf, tcf, inercia_transversal, inercia_longitunidal, LCB, TCB, KB, BML, BMT, desloc)


#------------------------------------------------------------------------------------#    

#Aproximando a nossa tabela de Cotas
def retorna_calculos_ep(nome_arquivo, num_interpolations):
    num_interpolations = int(num_interpolations)
    tabelaCotas=le_arquivo(nome_arquivo)
    #Fazendo quatro interpolções para cada direação. Após este número de interpolações não há mudanças significativas nos valores hidroestático
    tabelaCotas=SplineMultipla(tabelaCotas, num_interpolations)
    resultsPerCalado=[]
    inputs=np.arange(2,20,1)
    list_calados = []
    for i in range(len(inputs)):
        resultsCalado=picker(tabelaCotas, inputs[i])
        caladoIndex, caladoTab = resultsCalado[0], resultsCalado[1]
        #hidroPropshttp://localhost:8888/notebooks/PNV3315/arquivos_base/Ep%20-%20Hidrost%C3%A1tica.ipynb#
        resultsPerCalado.append(wrapper_infos(tabelaCotas, caladoIndex, caladoTab))
        list_calados.append(resultsCalado[1])
    data=[]
    inputs=inputs.tolist()
    parameters=["Área da superfície molhada (Sw) [m2]","Área do plano de linha d'água (Awl) [m2]","Vol Desloc. [m3]","LCF [m]","TCF [m]","IT [m4]",
                "IL [m4]","LCB [m]","TCB [m]","KB [m]","BML [m]","BMT [m]","Desloc. (Δ) [t]"]
    retorna_planilha_dados = pd.DataFrame(resultsPerCalado)
    retorna_planilha_dados.columns = parameters
    retorna_planilha_dados  = retorna_planilha_dados.T
    retorna_planilha_dados.columns = list_calados
    retorna_planilha_dados.to_excel('Resultados_hidrostáticos.xlsx') 
    return 'Arquivo gerado: Resultados_hidrostáticos.xlsx'

nome_arquivo = input('Nome arquivo: ')
file = nome_arquivo + '.xlsx'
num_interpolations = input('Quantas interpolações? ')
retorna_calculos_ep(file, num_interpolations)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




