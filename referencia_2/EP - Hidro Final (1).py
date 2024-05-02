#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms


# In[2]:


#Quantidade de pontos criados entre dois pontos iniciais 
n = int(input("Digite n: "))
qtd = 20*n


# In[3]:


#Leitura do plano de cotas, balizas na vertical e linhas d'agua na horizontal
basetemp = pd.read_excel(r'C:\Users\gusta\Desktop\Poli\Hidro\Spline.xlsx') #Colocar o diretório da tabela de cotas


# In[4]:

#Colocar coluna de texto dos S.T's como index
basetemp = basetemp.set_index('Unnamed: 0')


# In[5]:


#Visualização da tabela
base = basetemp.iloc[1:,:]
base.iloc[:, 1:] = base.iloc[:, 1:].multiply(22.98/2)
#Multiplicando pela depht (14.22)
base.columns = ["ST", "WL 0", "WL 0.711", "WL 1.422", "WL 2.844", "WL 4.266", "WL 5.688", "WL 7.11", "WL 8.532", "WL 9.954", "WL 11.376", "WL 14.22"]
#Multiplicando pelo comprimento
base["ST"] = base["ST"].multiply(6.275)



# # Para gerar splines

# In[7]:


# Gerar matriz A
def Calc_g(x, y):
    n = len(x)
    h = [0]
    I = np.zeros((n,n))
    I[0][0] = 1
    I[n-1][n-1] = 1
    
    for i in range(1,n):
        H = x[i]-x[i-1]
        h.append(H)

    h.append(0)

    for i in range (1,n-1):
        if i < (n-2):
            I[i][i-1] = h[i]
            I[i][i] = 2*(h[i]+h[i+1])
            I[i][i+1] = h[i+1]
        else:
            I[i][i-1] = h[i]
            I[i][i] = 2*(h[i]+h[i+1])
            I[i][i+1] = h[i+1]
            
    n = len(y)
    b = np.zeros((n,1))

    for i in range(1, n-1):

        b[i][0] = ((y[i+1]-y[i])/h[i+1]) - ((y[i] - y[i-1])/h[i])
        
    b *= 3
    g = np.matmul(np.linalg.inv(I), b)
    
    h_f = h[1:-1]
    
    return h_f, g


# In[8]:


def Calc_b(x, y):
    h, g = Calc_g(x,y)
    n = len(x)
    b = []
    
    for i in range (n-1):
        B = (1/h[i])*(y[i+1] - y[i]) -(h[i]/3)*(2*g[i]+g[i+1])
        b.append(float(B))
    return b


# In[9]:


def Calc_d(x,y):
    h, g = Calc_g(x,y)
    n = len(x)
    d = []
    
    for i in range(n-1):
        D = (g[i+1]-g[i])/(3*h[i])
        d.append(float(D))
    return d


# In[10]:


# Main de Spline (junta as outras)
def Spline(x,y):
    x_lista = []
    y_lista = []

    h, g = Calc_g(x,y)
    b = Calc_b(x,y)
    d = Calc_d(x,y)

    for i in range (len(x)-1):
        if i == len(x)-2:
            #Espaço linear criando pontos intermediários
            x_temp = np.linspace(x[i], (x[i+1]), n)
        else:
            x_temp = np.linspace(x[i], (x[i+1]), n, endpoint = False) # mudar aqui o número de pontos dentro de cada ponto
        
        for j in (x_temp):
            y_temp = y[i] + b[i]*(j - x[i]) + g[i]*((j - x[i])**2) + d[i]*((j - x[i])**3)
            y_lista.append(y_temp)

        x_lista.extend(x_temp)
        
    return x_lista, y_lista


# In[11]:


#Nova tabela para não modificar a original, com utilizaçao de splines -> balizas e meia boca intermediarias
df_novo = pd.DataFrame()
df_novo['B'] = np.linspace(0, 20, num = qtd)
df_novo["B"] = df_novo["B"].multiply(6.275)
for i in base.columns[1:]:
    x_points = base['ST']
    y_points = (base[i])
    x, y = Spline(x_points, y_points)
    
    df_novo['WL ' + str(i)] = y #Gera coluna nova



# In[12]:


#Tabela splinada -> linhas d'aguas calculada a partir da meia boca intermediaria
df_novo2 = pd.DataFrame()

df_novo2['B'] = np.linspace(0, 20, num = qtd)
df_novo2["B"] = df_novo2["B"].multiply(6.275)

WLs = df_novo.columns[1:].str.replace("WL ", "").astype(float)
x_points = WLs
r = []
for i, row in df_novo2.iterrows():

    y_points = (df_novo.iloc[i,1:])
    
    x, y = Spline(x_points, y_points)
    
    for j in x:
        df_novo2['WL ' + str(float(j))] = 0.
    
    r.append(pd.Series(y).astype(float).T.values)
    
for i in range(len(df_novo2.columns)-1):
    df_novo2.iloc[:, i+1] = np.transpose(r)[i]


# In[14]:


#Gráfico das balizas pela meia boca com linhas intermediárias
plt.figure(facecolor=".9", figsize = (40,7))
x_points = df_novo2['B']

for i in df_novo2.columns[1:]:
    
    y_points = (df_novo2[i])
    plt.plot(x_points, y_points, color = 'navy')
    
plt.show()


# In[15]:

#meia boca x linhas d'agua -> nao da pra splinar quando x1 = x2 (calculou deitado e colocou em pé -> rotacionou ) -> c/ linha d'agua
plt.figure(facecolor=".9", figsize = (10,10))
z_points = df_novo2.columns[1:].str.replace("WL ", "").astype(float) #Linhas d"água
for i in range (int((len(df_novo2)/2)+1)): #Para aft stations
    
    basex = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    
    y_points = df_novo2.iloc[i,1:]
    z, y1 = Spline(z_points, y_points)
    plt.plot(z, y1, transform = rot + basex, color = 'navy') 
    
for i in range (int((len(df_novo2)/2)+1),(len(df_novo2))): #Para foward stations
    
    basex = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    
    y_points = -df_novo2.iloc[i,1:] #multiplicação por -1 para ficar espelhado
    z, y1 = Spline(z_points, y_points)
    plt.plot(z, y1, transform = rot + basex, color = 'navy')
plt.show()


# In[17]:


#cria painel e calcula as áreas associadas aos paineis
#Separando em ida e volta (Gera duas áreas - dividindo obtêm-se a intersecção das áreas - melhor aproximação)
def ObterA(df_novo2, h): 
    
    VetoresW = []
    for j in df_novo2.columns[1:h-1]:
        Vetores_Temp = []
        for i in range(1, len(df_novo2[j])):
            vety = df_novo2[j][i] - df_novo2[j][i-1]
            vetb = df_novo2["B"][i] - df_novo2["B"][i-1]
            Vetores_Temp.append([vetb, vety, 0])
        VetoresW.append(np.array(Vetores_Temp))
    

    VetoresB = []
    for j in range(len(df_novo2)-1):
        Vetores_Temp = []
        for i in range(2, h):
            vety = df_novo2.iloc[j, :][i] - df_novo2.iloc[j, :][i-1]
            vetw = float(df_novo2.columns[i].replace("WL ", "")) - float(df_novo2.columns[i-1].replace("WL ", ""))
            Vetores_Temp.append([0, vety, vetw])
        VetoresB.append(np.array(Vetores_Temp))
    
    idatemp = []
    ida = []
    for i in range (len(VetoresW)):
        for j in range (len(VetoresB)):
            res = np.cross( VetoresB[j][i], VetoresW[i][j])
            idatemp.append(res)
        ida.append(idatemp)
        idatemp =[]

    VetoresW2 = []
    for j in df_novo2.columns[h-1:1:-1]:
        Vetores_Temp = []
        for i in range(len(df_novo2.iloc[:, 5])-1, 0, -1):
            vety = df_novo2[j][i-1] - df_novo2[j][i]
            vetb = df_novo2["B"][i-1] - df_novo2["B"][i]
            Vetores_Temp.append([vetb, vety, 0])
        VetoresW2.append(np.array(Vetores_Temp))


    VetoresB2 = []
    for j in range(len(df_novo2.iloc[:, 0]) - 1, 0, -1):
        Vetores_Temp = []
        for i in range(h - 1, 1, -1):
            vety = df_novo2.iloc[j, :][i-1] - df_novo2.iloc[j, :][i]
            vetw = float(df_novo2.columns[i-1].replace("WL ", "")) - float(df_novo2.columns[i].replace("WL ", ""))
            Vetores_Temp.append([0, vety, vetw])
        VetoresB2.append(np.array(Vetores_Temp))

    voltatemp = []
    volta = []
    for i in range (len(VetoresW2)):
        for j in range (len(VetoresB2)):
            res = np.cross(VetoresB2[j][i], VetoresW2[i][j])
            voltatemp.append(res)
        volta.append(voltatemp)
        voltatemp=[]

    Af = (np.array(ida) + list(reversed(np.array(volta))))/2.
    return Af

Area = np.array([])
Calado = []
AM = []
Z = []
for i in range (len(df_novo2.columns),2,-1):

    Af = ObterA(df_novo2, i)
    Z.append(np.sum(Af[i-3, :,-1]))
    for j in Af:
        A = np.linalg.norm(j)
        Area = np.append(Area, A)
    AM.append(np.sum(Area))
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    Area = np.array([])

# n = 75
# Tempo -> 50 min

# In[41]:


Area = np.array([])
Calado = []
AM = []
AWl = []
Z = np.array([])
for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)

    for j in Af:
        for k in j:
            z = -k[-1]
            Z = np.append(Z, z)
    for j in Af:
        for k in j:
            A = np.linalg.norm(k)
            Area = np.append(Area, A)
    AWl.append(np.sum(Z))
    AM.append(np.sum(Area))
    
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    Area = np.array([])
    Z = np.array([])

# Para Área plano da linha d'água (AWl)
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(AWl, Calado, color = "navy")
plt.xlabel("Área plano de linha d'água (m^2)") 
plt.ylabel('Calado (m)')
plt.title("Área WL x Calado")
plt.show()
plt.clf()

#Para Área submersa (AM)
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(AM, Calado, color = "navy")
plt.xlabel("Área molhada (m^2)") 
plt.ylabel('Calado (m)')
plt.title("Área molhada x Calado")
plt.show()


# In[19]:


#Área molhada máxima (AMt)
AMt = 0
Af = ObterA(df_novo2, len(df_novo2.columns)) 

k = 0
for j in Af:
    for i in j:
        A = np.linalg.norm(i)
        AMt += A
        k+=1


# In[42]:


Calado = []
cima = []
baixo = []
LCF = np.array([])

Cx = []
for i in range(1, len(df_novo2["B"])):
    C_x = (df_novo2["B"][i-1] + df_novo2["B"][i])/2
    Cx.append(C_x)
    

for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    for j in range(len(Af)):
        ctemp = Cx[j]
        for k in Af[j]:
            z = k[-1]
            cima.append((z*ctemp))
            baixo.append(z)

    LCF = np.append(LCF, np.sum(cima)/np.sum(baixo))
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    cima = []
    baixo = []

#Para LCF
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(LCF, Calado, color = "navy")
plt.xlabel("LCF (m)") 
plt.ylabel('Calado (m)')
plt.title("LCF x Calado")
plt.show()


# In[21]:


#Calculo dos centroides (Cx, Cy e Cz)
Calado = []
somaC = []

Il = np.array([])

Cx = []
for i in range(1, len(df_novo2["B"])):
    C_x = (df_novo2["B"][i-1] + df_novo2["B"][i])/2
    Cx.append(C_x)
    
Cy = []
for i in range(1, len(df_novo2.columns)-1):
    Ctemp = []
    for j in range (1, len(df_novo2["B"])):
        C_y = (df_novo2.iloc[j-1, i] + df_novo2.iloc[j, i])/2
        Ctemp.append(C_y)
    Cy.append(Ctemp)
    
Cz = []
for i in range(2, len(df_novo2.columns)):
    C_z = (float(df_novo2.columns[i-1].replace("WL ", "")) + float(df_novo2.columns[i].replace("WL ", "")))/2
    Cz.append(C_z)


# In[22]:


#Para calculo do volume (Base volume Cy)
def Volume(Cy, calado, df_novo2):
    Af = ObterA(df_novo2, calado)

    Ay = np.array([])
    for j in range(len(Af)):
        for m in range(len(Af[j])):
            AreaY = np.array(Cy[j][m]) * Af[j,m,1]
            Ay = np.append(Ay, AreaY)

    Fy = np.sum(Ay)
    return Fy
Volume(Cy, len(df_novo2.columns), df_novo2) #volume máximo, embarcação pela metade


# In[43]:


#Calculo BMl
listaI = []
BMl = []
Calado = []

for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    LCFtemp = LCF[i-3]
    for j in range(len(Af)):
        I = -(Af[j,:,-1]*(np.square(Cx - LCFtemp)))
        listaI.append(I)
    Itemp = np.sum(np.array(listaI))
    V = Volume(Cy, i, df_novo2)
    BMl.append(Itemp/V)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    listaI = []
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(BMl, Calado, color = "navy")
plt.xlabel("BML (m)") 
plt.ylabel('Calado (m)')
plt.title("BML x Calado")
plt.show()


# In[44]:


#Momento para trimar
listaI = []
Calado = []
MT = []
for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    LCFtemp = LCF[i-3]
    for j in range(len(Af)):
        I = -(Af[j,:,-1]*(np.square(Cx - LCFtemp)))
        listaI.append(I)
    Itemp = np.sum(np.array(listaI))
    V = Volume(Cy, i, df_novo2)
    temp = (BMl[i-3]*(V*10.0616))/12550
    MT.append(temp)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    listaI = []
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(MT, Calado, color = "navy")
plt.xlabel("MT (t x cm)") 
plt.ylabel('Calado (m)')
plt.title("MT x Calado")
plt.show()


# In[45]:


#BMT
listaI = []
BMt = []
Calado = []

for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    for j in range(len(Af)):
        I = -(Af[j,:,-1]*(np.square(Cy)))
        listaI.append(I)
    Itemp = np.sum(np.array(listaI))
    V = Volume(Cy, i, df_novo2)
    BMt.append(Itemp/V)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    listaI = []
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(BMt, Calado, color = "navy")
plt.xlabel("BMt (m)") 
plt.ylabel('Calado (m)')
plt.title("BMT x Calado")
plt.show()


# In[46]:


#LCB
listaI = []
LCB = []
Calado = []

for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    for j in range(len(Af)):
        I = np.multiply(Af[j,:,0],Cx) *np.array(Cx)/2
        listaI.append(I)
    Itemp = np.sum(np.array(listaI))
    V = Volume(Cy, i, df_novo2)
    LCB.append(Itemp/V)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    listaI = []
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(LCB, Calado, color = "navy")
plt.xlabel("LCB (m)") 
plt.ylabel('Calado (m)')
plt.title("LCB x Calado")
plt.show()


# In[27]:


#KB
listaI = []
KB = []
Calado = []
AZ = []

for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    for k in range (len(Af)):
        AZ.append(np.sum(Af[k,:,2]))
    
    Cz = []
    for i in range(2, len(Af)+2):
        C_z = (float(df_novo2.columns[i-1].replace("WL ", "")) + float(df_novo2.columns[i].replace("WL ", "")))/2
        Cz.append(C_z)
    
    for j in range(len(Af)):
        I = -np.multiply(AZ,Cz) *np.array(Cz)/2
        listaI.append(np.sum(I))
    Itemp = np.sum(np.array(listaI))
    V = Volume(Cy, i, df_novo2)
    KB.append(Itemp/V)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    AZ = []
    ListaI = []
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(KB, Calado, color = "navy")
plt.xlabel("KB (m)") 
plt.ylabel('Calado (m)')
plt.title("KB x Calado")
plt.show()


# In[38]:


#Volume x Calado
Vol = []
Calado = []

for i in range (len(df_novo2.columns),2,-1):
    V = Volume(Cy, i, df_novo2)
    Vol.append(V)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))

plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(Vol, Calado, color = "navy")
plt.xlabel("Volume (m³)") 
plt.ylabel('Calado (m)')
plt.title("Volume x Calado")
plt.show()


# In[39]:


#Deslocamento x Calado
Desloc = []
Calado = []

for i in range (len(df_novo2.columns),2,-1):
    V = Volume(Cy, i, df_novo2)
    Desloc.append(V/1.026)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))

plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(Desloc, Calado, color = "navy")
plt.xlabel("Deslocamento (t)") 
plt.ylabel('Calado (m)')
plt.title("Deslocamento x Calado")
plt.show()


# In[37]:


#Inércia longitudinal x Calado
listaI = []
InérciaL = []
Calado = []

for i in range (len(df_novo2.columns),2,-1):
    Af = ObterA(df_novo2, i)
    LCFtemp = LCF[i-3]
    for j in range(len(Af)):
        I = -(Af[j,:,-1]*(np.square(Cx - LCFtemp)))
        listaI.append(I)
    Itemp = np.sum(np.array(listaI))
    InérciaL.append(Itemp)
    Calado.append(float(df_novo2.columns[i-1].replace("WL ", "")))
    listaI = []
plt.figure(facecolor=".9", figsize = (12,6))
plt.plot(InérciaL, Calado, color = "navy")
plt.xlabel("Inércia Longitudinal(m^4)") 
plt.ylabel('Calado(m)')
plt.title("Inércia Longitudinal x Calado")
plt.show()


# In[32]:

#Plotando todos gráficos juntos

plt.figure(facecolor=".9", figsize = (12,6))
# Para Área plano da linha d'água
plt.plot((np.array(AWl)/10.), Calado, label = "Área plano de linha d'água (10m²)")

#Para Área submersa
plt.plot((np.array(AM)/20.), Calado, label = "Área molhada (20m²)")

#Para Área submersa
plt.plot(LCF, Calado, label = "LCF (m)")

#Para Volume
plt.plot((np.array(Vol)/500.), Calado, label = "Volume (500m³)")

#Para Deslocamento
plt.plot((np.array(Desloc)/500.), Calado, label = "Deslocamento (500t)")

#Para BML
plt.plot((np.array(BMl)/20.), Calado, label = "BML (20m)")

#Para Inércia Longitudinal
plt.plot((np.array(InérciaL)/1e+6), Calado, label = "Inércia Longitudinal (1e+6m^4)")

#Para MT
plt.plot((np.array(MT)/200.), Calado, label = "MT (200t/cm)")

#Para BMT
plt.plot((np.array(BMt)/20.), Calado, label = "BMT (20m)")

plt.ylabel('Calado (m)')
plt.title("Propriedades hidrostáticas x Calado")
plt.legend(title = 'Propriedades', bbox_to_anchor=(0., -.37, 1., -.102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.grid()
plt.show()


