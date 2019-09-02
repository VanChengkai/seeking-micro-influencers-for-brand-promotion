import xlrd
import numpy as np

ExcelFile1=xlrd.open_workbook('G:\\aaa\\test_set.xlsx')
sheet1=ExcelFile1.sheet_by_index(0)
l_user=[]
l_in=[]
l_ist=[]
l_score=[]

def metrics(l_user,l_in,l_ist,l_score):#MedR,recall@10,recall@50
    all_positive=790
    # test set has 790 positive examples
    part=len(l_user)/all_positive
    part=int(part)
    index=0
    index2=0
    lll=[]
    lll1=[]
    lll2=[]
    
    lis=[]
    for k in range(0,all_positive):
        lis.append(k+1)
    for j in range(0,part):
        print(j)
        l=[]
        for i in range(0,all_positive):
            a=l_score[index]
            l.append(a)
            index+=1
            if((i+1)%all_positive ==0 or index == len(l_user)):
                break
            
        l.sort()
        ll=[]
        #shu=0
        for i in range(0,all_positive):
            
            user=l_user[index2]
            in_=l_in[index2]
            ist=l_ist[index2]
            score=l_score[index2]
            paiming=l.index(score)
            
            ist=int(ist)
           
            
            if(ist==1):
                
                ll.append(all_positive-paiming)
                
            if((i+1)%all_positive==0 or index2 == len(l_user)):
                
                print(ll)
                y=min(ll)
                lll2.append(y)
                index1=0
                index3=0
                for a in ll:
                    if(a<11):
                        index1+=1
                    if(a<51):
                        index3+=1
                p1=index1/len(ll)
                p2=index3/len(ll)
                
                lll.append(p1)
                lll1.append(p2)
                
                break
            index2+=1
    


    al=0.0
    lll2.sort()
    med=0
    if(len(lll2)%2==0):
        med=len(lll2)/2
    else:
        med=int(len(lll2)/2)+1
    for xy in range(0,len(lll2)):
        if(xy==(med-1)):
            al=lll2[xy]
    #MedR
    print(al)
    
    al2=0.0
    for xy1 in lll:
        
        al2+=xy1
    #recall@10
    print(al2/len(lll))
    
    al3=0.0
    for xy2 in lll1:
        
        al3+=xy2
    #recall@50
    print(al3/len(lll1))
    
    
def auc(l_user,l_in,l_ist,l_score):
    ExcelFile2=xlrd.open_workbook('G:\\aaa\\test_set_auc_v3.xlsx')
    sheet2=ExcelFile2.sheet_by_index(0)
    err=0
    AUC=0.0
    AUC_all=0.0
    cAUC=0.0
    cAUC_all=0.0
    
    dict_s={}
    
    for i in range(0,len(l_user)):
        a_=l_user[i]
        b_=l_in[i]
        score1=l_score[i]
        a=a_+b_
        dict_s[a]=score1
    for i in range(0,sheet1.nrows):
        if(i%10000==0):
            print(i)
        AUC_all+=1
        a=sheet2.cell(i,0).value.encode('utf-8').decode('utf-8-sig')
        b=sheet2.cell(i,1).value.encode('utf-8').decode('utf-8-sig')
        c=sheet2.cell(i,3).value.encode('utf-8').decode('utf-8-sig')
        d=sheet2.cell(i,4).value.encode('utf-8').decode('utf-8-sig')
        e=sheet2.cell(i,5).value
        score1=0.0
        score2=0.0
        if a+b in dict_s.keys():
            score1=dict_s[a+b]
        if c+d in dict_s.keys():
            score2=dict_s[c+d]
        
        
        if(score1==0.0 or score2==0.0):
            err+=1
        if(e!=0):
            cAUC_all+=1
        if(score1>score2):
            AUC+=1
            if(e!=0):
                #print(score1,score2)
                cAUC+=1
                
    print('AUC is:',AUC/AUC_all,AUC)
    print('cAUC is:',cAUC/cAUC_all,cAUC)
    print(err)

if __name__ == '__main__':
    #data have divided into 6 parts
    for k in range(0,6):
        print('loading...test')
        path1='F:\\dataset_k_3_test\\brand_text_test_'+str(k)+'.npy'
        brand_text_test=np.load(path1)
        print('loading...test')
        in_text_test=np.load('F:\\dataset_k_3_test\\in_text_test_'+str(k)+'.npy')
        print('loading...test')
        brand_pic_test=np.load('F:\\dataset_k_3_test\\brand_pic_test_'+str(k)+'.npy')
        print('loading...test')
        in_pic_test=np.load('F:\\dataset_k_3_test\\in_pic_test_'+str(k)+'.npy')
        for i in range(0,len(brand_text_test)):
            
            a1=brand_text_test[i].tolist()
            a2=in_text_test[i].tolist()
           
            b=[]
            b1=brand_pic_test[i].tolist()
            b2=in_pic_test[i].tolist()
            b1.extend(a1)
            b2.extend(a2)
            sb=0.0
            for kk in range(0,len(b1)):
                sb+=(b1[kk]*b2[kk])
            sb=sb/len(b1)
            l_score.append(sb)
        
    for j in range(0,sheet1.nrows):
        user=sheet1.cell(j,0).value.encode('utf-8').decode('utf-8-sig')
        in_=sheet1.cell(j,1).value.encode('utf-8').decode('utf-8-sig')
        ist=sheet1.cell(j,2).value
        l_user.append(user)
        l_in.append(in_)
        l_ist.append(ist)
        
        
    metrics(l_user,l_in,l_ist,l_score)
    auc(l_user,l_in,l_ist,l_score)
