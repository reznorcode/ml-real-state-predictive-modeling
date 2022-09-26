import numpy as np
import pandas as pd

import os

import re

import pickle

import warnings
warnings.filterwarnings( 'ignore' )

from class_previsao import Precificacao

if __name__ == '__main__':
        
    precificacao = Precificacao()

    df_raw = precificacao.ler_csv('consolidadoV3.csv')

    df = precificacao.data_rearrangement( df_raw)


    df1 =precificacao.data_transform( df)


    df1_clean = precificacao.data_filter(df1) 


    df1_join_precom2 = precificacao.df1_merge_preco_metro('Bairro_valor_metro2.csv',df1_clean ) 

    df2 = precificacao.feature_engineering( df1_join_precom2 ) 

    selection = [
    'Banheiros',
    'Vagas',
    'Bairro',
    'preco_por_metro',
    'QuartosporAreaConstruida',
    'LogAreaConstruida',
    'bairro_por_area'
    ]

    df2_sel = df2[selection]

    df3 = precificacao.data_preparation(df2_sel)


    test= precificacao.ler_csv('test_prev_sing.csv')
    test_data = precificacao.data_preparation(test)

    test_data.drop('Unnamed: 0', axis=1, inplace=True) 

    prev = round(precificacao.get_prediction(test_data)[0], 2) 

    print(prev)
    print('Maior preço', prev + 155538.56 )
    print('Menor preço', prev - 35915.09 )

    print(' REAL')
    print(df2[df2.Banheiros == 2][df2.Vagas == 2][df2.Bairro == 'Recanto das Palmeiras'][df2.Quartos == 3][df2['Area construida'] >= 60])


