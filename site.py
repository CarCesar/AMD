import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import  Ridge, RidgeCV, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor


def colorirVED(s):
    if s[-1] <5:
        return ['background-color: #AAFFBB']*7
    elif s[-1] > 16:
        return ['background-color: #FFAABB']*7
    else:
        return ['background-color: #FFFFCC']*7

#3DEF2B
#DE2B02
#E5E90C

def colorirPT(s):
    if s[-1] <5:
        return ['background-color: #AAFFBB']*3
    elif s[-1] > 16:
        return ['background-color: #FFAABB']*3
    else:
        return ['background-color: #FFFFCC']*3

def DADOS(url:str,n_rodadas):
    # Pega o DF e faz a media para as categorias que são cabiveis de tal ação.
    df = pd.read_csv(url,sep = ';')
    df = df.drop(['VCasa',"ECasa","DCasa","GPCasa","GCCasa",
                  "PtCasa","VFora","EFora","DFora","GPFora",
                  "GCFora","PtFora"],axis=1)
    df.iloc[:,4:37]=df.iloc[:,4:37]/n_rodadas
    df.iloc[:,2]=df.iloc[:,2]/10
    return df

def reduzir(df,coluna,theta):
    #Pega as colunas mais importantes para os dados...
    model = RandomForestRegressor(random_state=1, max_depth=10)
    model.fit(df.iloc[:,1:37],df.loc[:,coluna])
    features = df.columns[1:37]
    importances = model.feature_importances_
    indices = np.argsort(importances) #[0:9])  # top 10 features
    new_model = SelectFromModel(model,threshold=theta)
    fit = new_model.fit_transform(df.iloc[:,1:37], df.loc[:,coluna])
    new_model.get_support()
    selected_columns = np.array(df.columns[1:37])[new_model.get_support()]
    df3 = df[selected_columns]
    pd.options.mode.chained_assignment = None
    df3[coluna] = df[coluna]
    return df3

def tablePT(url1:str,url2:str,n_rodadas:int,TESTE:bool,MODELO:str):
    treino = DADOS(url1,38)
    teste = DADOS(url2,n_rodadas)
    scaler = MaxAbsScaler().fit(treino.iloc[:,1:37])
    a = scaler.transform(treino.iloc[:,1:37])
    treino.iloc[:,1:37]=a
    reduzido = reduzir(treino,'Pt',0.05)
    X = reduzido.values[:,0:-1]
    Y = reduzido.values[:,-1]
    if MODELO == 'RidgeCV':
        modelo= RidgeCV().fit(X,Y)
    if MODELO == 'BaysianRidge':
        modelo = BayesianRidge(n_iter = 2000).fit(X,Y)
    if MODELO == 'ExtraTreesRegressor':
        modelo = ExtraTreesRegressor(random_state=0).fit(X,Y)
    teste.iloc[:,1:37] = scaler.transform(teste.iloc[:,1:37])
    prev = modelo.predict(teste[reduzido.columns].values[:,0:-1])
    r = pd.DataFrame()
    r['Equipe']= teste.Equipe
    r['PP']= prev
    r=r.sort_values('PP', ascending=False)
    if TESTE:
        r['Pontos'] = teste.Pt
        r['CP']=range(1,21)
        r['CR']= teste.Posicao
        r['diff'] = abs(r['CP'] -r['CR'])
        r.set_index('Equipe', inplace=True)
    else:
        r['CP']=range(1,21)
        r.index = r.CP
        r=r.style.apply(colorirPT, axis=1)
    return(r)

def tableVED(url1:str,url2:str,n_rodadas:int,TESTE:bool,MODELO:str):
    treino = DADOS(url1,38)
    teste = DADOS(url2,n_rodadas)
    scaler = MaxAbsScaler().fit(treino.iloc[:,1:37])
    a = scaler.transform(treino.iloc[:,1:37])
    treino.iloc[:,1:37]=a
    teste.iloc[:,1:37] = scaler.transform(teste.iloc[:,1:37])
    dfV = reduzir(treino,'V',0.05)
    dfE = reduzir(treino,'E',0.003)
    dfD = reduzir(treino,'D',0.05)
    Xv = dfV.values[:,0:-1]
    Yv = dfV.values[:,-1]
    Xe = dfE.values[:,0:-1]
    Ye = dfE.values[:,-1]
    Xd = dfD.values[:,0:-1]
    Yd = dfD.values[:,-1]
    if MODELO == 'RidgeCV':
        V = RidgeCV().fit(Xv,Yv)
        E = RidgeCV().fit(Xe,Ye)
        D = RidgeCV().fit(Xd,Yd)
    if MODELO == 'BaysianRidge':
        V = BayesianRidge(n_iter = 2000).fit(Xv,Yv)
        E = BayesianRidge(n_iter = 2000).fit(Xe,Ye)
        D = BayesianRidge(n_iter = 2000).fit(Xd,Yd)
    v = teste[dfV.columns]
    vprev = V.predict(v.values[:,0:-1])
    e = teste[dfE.columns]
    eprev = E.predict(e.values[:,0:-1])
    d = teste[dfD.columns]
    dprev = D.predict(d.values[:,0:-1])
    w = pd.DataFrame()
    w['Equipe'] = teste.Equipe
    w['VP'] = vprev
    w['EP'] = eprev
    w['DP'] = dprev
    w['rodadas'] = w.VP+w.EP+w.DP
    w.VP = np.round(w.VP/w.rodadas*38).astype('int')
    w.EP = np.round(w.EP/w.rodadas*38).astype('int')
    w.DP = np.round(w.DP/w.rodadas*38).astype('int')
    w['rodadas'] = w.VP+w.EP+w.DP
    w.EP[w.rodadas == 37] = w.EP +1 
    w.EP[w.rodadas == 39] = w.EP -1 
    w['rodadas'] = w.VP+w.EP+w.DP
    w['PP'] = w.VP * 3 + w.EP
    w=w.sort_values('PP', ascending=False)
    if TESTE:
        w['pontos'] = teste.Pt
        w['CP']=range(1,21)
        w['CR']=teste.Posicao
        w['diff'] = abs(w.CP-w.CR)
        w.set_index('Equipe', inplace=True)
    else:
        w['CP']=range(1,21)
        w.index = w.CP
        #w=w.style.background_gradient(axis=None,cmap='RdYlGn_r', subset=['i'])
        w=w.style.apply(colorirVED, axis=1)
    return(w)


f1,b,f2= st.sidebar.columns([9,1,10])
f1.markdown('**Emap - FGV**')
f2.markdown('**Novembro 2021**')
texto = '''
# Trabalho de ADM

Para esse trabalho fizemos:
- Web-Scraping dos dados, [daqui](https://fbref.com/pt/comps/38/Serie-B-Estatisticas)
- Etapas do processo no [Github](https://github.com/CarCesar/AMD)

Fomos **insanos** na hora da escolha do trabalho, e nossa ideia era **prever algo complexo** de uma **forma simples**. Apesar disso, fizemos um trabalho legal... 


##### Quem fez?
'''

st.sidebar.markdown(texto)
b,sc1,b,sc2,b=st.sidebar.columns([1,4,1,4,1])
sc1.image('andre.png',caption='André Costa')
sc2.image('carlos2.png',caption = 'Carlos César')

#st.sidebar.markdown('-----')


b1,b,b2=st.columns([15,1,13])
b1.title('Previsão SERIE B')
b1.markdown('''Previsão de um jeito maluco,as vezes, inlógico, da classificação final da SERIE B

''')


menu = b1.radio('Faça sua escolha',('Teste','Pratica'),index = 1)
b2.write('\n')
#b2.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKHxnQQjFmUddwNTLlcZzuJwgsUE_hLrO3VA&usqp=CAU')
b2.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRC8tvxY0cVAiyCMMUJzkmmnZC6LJ8Fq5lSOg&usqp=CAU', caption = 'Saudade de 2017...',width = 300)

tteste="""# Teste
Não sabemos se foi a melhor forma de trabalhar com os dados, porém achamos legais os resultados.

- Treinamos com o **campeonato de 2019**
- Testamos com o **campeonato de 2020**

Escolha qual tipo de tabela com qual modelo quer ver...
                    """

tpratica = '''# Prática

Aqui você deve escolher: 
- se prevê:
    - o numero de pontos(PT) 
    - o numero de V/E/D para ter os pontos(VED)
- um modelo do scikit learn (selecionamos alguns p/ cada tipo)
- o ano campeonato que o modelo usará para treinar (2020, 2019 , 2019 e 2020)
- após qual rodada pegamos os dados do campeonato de 2021

E então verá a tabela conforme nosso modelo previu
'''

t = ['VP - vitórias previstas', 'EP - empates previstos', 'DP - derrotas previstas',
     'PP - pontos previstos', 'CP - classificação prevista', 'CR - classificação real']
sp = ' // '

link = 'https:\\raw.githubusercontent.com\CarCesar\AMD\\main'

def sidebar(MENU):
    if MENU == 'Teste':
        st.markdown(tteste)
        colum,coldo=st.columns(2)
        option = colum.selectbox('Escolha:',('','PT','VED'))
        if option =='':
            modelinho = coldo.selectbox('Modelo:',())
        if option =='PT':
            modelinho = coldo.selectbox('Modelo:',('','RidgeCV','BaysianRidge','ExtraTreesRegressor'))
        if option =='VED':
            modelinho = coldo.selectbox('Modelo:',('','RidgeCV','BaysianRidge'))
        if modelinho != '':
            if option == 'VED':
                st.table(tableVED('dados2019.csv','dados2020.csv',38,True,modelinho).style.background_gradient(axis=None,cmap='RdYlGn_r', subset=['diff']))
                st.write('**Legenda : **'+t[0]+sp+t[1]+sp+t[2]+sp+t[3]+sp+t[4]+sp+t[5])
            if option == 'PT':
                st.table(tablePT('dados2019.csv','dados2020.csv',38,True,modelinho).style.background_gradient(axis=None,cmap='RdYlGn_r', subset=['diff']))
                st.write('**Legenda : **'+t[3]+sp+t[4]+sp+t[5])
    elif MENU == "Pratica":
        st.markdown(tpratica)
        #tudo = st.checkbox('Colocar todos os anos')
        col1,col2,col3,col4=st.columns(4)
        option = col1.selectbox('Tipo:',('','PT','VED'))
        if option =='':
            modelinho = col2.selectbox('Modelo:',())
        if option =='PT':
            modelinho = col2.selectbox('Modelo:',('','RidgeCV','BaysianRidge','ExtraTreesRegressor'))
        if option =='VED':
            modelinho = col2.selectbox('Modelo:',('','RidgeCV','BaysianRidge'))
        ano = col3.selectbox('Ano  de treino:',('2019-2020','2020','2019'))
        rodada = col4.selectbox('Rodada para pratica:',(33,35))
        if modelinho !='':
            if option == 'VED':
                if ano == '2019':
                    if rodada == 33:
                        st.table(tableVED('dados2019.csv','ate33.csv',33,False,modelinho))
                    if rodada == 35:
                        st.table(tableVED('dados2019.csv','ate35.csv',35,False,modelinho))
                if ano == '2020':
                    if rodada == 33:
                        st.table(tableVED('dados2020.csv','ate33.csv',33,False,modelinho))
                    if rodada == 35:
                        st.table(tableVED('dados2020.csv','ate35.csv',35,False,modelinho))
                if ano == '2019-2020':
                    if rodada == 33:
                        st.table(tableVED('oi1920.csv','ate33.csv',33,False,modelinho))
                    if rodada == 35:
                        st.table(tableVED('oi1920.csv','ate35.csv',35,False,modelinho))
                st.write('**Legenda : **'+t[0]+sp+t[1]+sp+t[2]+sp+t[3]+sp+t[4])
            if option == 'PT':
                #st.table(tablePT('.\Dados\dados2019.csv','.\Dados\\ate33.csv',33,False))
                if ano == '2019':
                    if rodada == 33:
                        st.table(tablePT('dados2019.csv','ate33.csv',33,False,modelinho))
                    if rodada == 35:
                        st.table(tablePT('dados2019.csv','ate35.csv',35,False,modelinho))
                if ano == '2020':
                    if rodada == 33:
                        st.table(tablePT('dados2020.csv','ate33.csv',33,False,modelinho))
                    if rodada == 35:
                        st.table(tablePT('dados2020.csv','ate35.csv',35,False,modelinho))
                if ano == '2019-2020':
                    if rodada == 33:
                        st.table(tablePT('oi1920.csv','ate33.csv',33,False,modelinho))
                    if rodada == 35:
                        st.table(tablePT('oi1920.csv','ate35.csv',35,False,modelinho))
                st.write('**Legenda : **'+t[3]+sp+t[4])


sidebar(menu)
