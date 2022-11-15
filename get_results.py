import os

data_name = 'alimama'
data_name = 'alicpp'
import pandas as pd

#best for norelue  1025,1027,1028
#must_keywords=['AutoInt_32_0.005']
#must_keywords=['Star_Trans_32_0.005']
#must_keywords=['Starv3_Trans_32_0.005_3_4_QK','_nometanorm']



if data_name=='alicpp':
    must_keywords=['WDL']
    must_keywords=['DeepFM_32']
    must_keywords=['DCN_32']
    must_keywords=['AutoInt_32_0.005']
    must_keywords=['Star_Net_32_0.005']

    must_keywords=['Starv3_Trans_32_0.005','QK','norelu,']
    #must_keywords=['Starv3_Trans_32_0.005','norelu','qkvid-noqkmap-sum,']
    #must_keywords=['Starv3_Tran','nometanorm']
    #must_keywords=['Meta_Trans_32_0.005','QK','mex,']
    #must_keywords=['Meta_Trans_v2','mex,']

    n_cols=3
    or_keywords = ['1027', '1028','1029','1030','1031']
    or_keywords = ['1023']
    or_keywords = []


if data_name=='alimama':
    must_keywords=['WDL']
    must_keywords=['DeepFM_32']
    must_keywords=['DCN_32']
    must_keywords=['AutoInt_32_0.001']
    must_keywords=['Star_Net_32_0.001']

    #must_keywords=['Starv3_Trans_32_0.001','norelu,']
    must_keywords=['Starv3_Trans_32_0.001','norelu','cat,']
    must_keywords=['Starv3_Tran','nometanorm']
    must_keywords=['Starv3_Trans_32_0.001','norelu','cat,']

    n_cols=2
    or_keywords = []

#or_keywords=['1025','1027','1028']#norelu
#or_keywords=['1025','1026','1027']#norelu

#must_keywords=['Starv3_Trans_32_0.005_3_4_QK','norelu-layerid-qkvid-noqkmap']


if os.path.exists(f'{data_name}_results.csv'):

    print('start')
    f=open(f'{data_name}_results.csv')
    results = f.readlines()

    outs = []
    for line in results:
        line=line.strip()
        flag=True
        for kw in must_keywords:
            if kw not in line:
                flag=False
        if len(or_keywords)==0:
            flag2=True
        else:
            flag2=False
            for kw in or_keywords:
                if kw in line:
                    flag2=True
        if flag and flag2:
            outs.append(line)

    outs = set(outs)
    df=[]
    for line in outs:
        x=line.split(',')
        for i in range(1,len(x)):
            x[i]=float(x[i])
        if len(x)==n_cols+2:
            df.append(x)
            print(line)
        else:
            #assert 1==0
            pass
            print(line)

    columns = ['flag','all']+[f'D_{i}' for i in range(n_cols)]


    df = pd.DataFrame(df,columns=columns)
    print(df[columns[1:]].mean())
