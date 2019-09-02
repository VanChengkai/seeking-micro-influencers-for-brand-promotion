import tensorflow as tf
import os
import xlrd
import time
from six.moves import xrange
import numpy as np
in_vector_size=2
brand_vector_size=2
labels_vector_size=1
batch_size=64
text_size=300
pic_size=25088
text_layer1_size=300
text_layer4_size=512
pic_layer1_size=4096
pic_layer4_size=512
model_path='/home/model6'
zu_size=4
def influencer_vectors_inputs():
    influencers_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         text_size))
    influencers_pic_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         pic_size))
    #labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return influencers_text_placeholder,influencers_pic_placeholder

def label_vector_inputs():
    #labels_placeholder = tf.placeholder(tf.float32, shape=(None, labels_vector_size))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None))
    return labels_placeholder
def brand_vector_inputs():
    brand_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         text_size))
    brand_pic_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         pic_size))
    return brand_text_placeholder,brand_pic_placeholder

def get_batch(brand_text_,brand_pic_,in_text_,in_pic_,labels_,step):
    if((step+1)*batch_size*zu_size>len(brand_text_)):
        brand_text=brand_text_[step*batch_size*zu_size:]
        brand_pic=brand_pic_[step*batch_size*zu_size:]
        in_text=in_text_[step*batch_size*zu_size:]
        in_pic=in_pic_[step*batch_size*zu_size:]
        label_=labels_[step*batch_size*zu_size:]
        label_ = label_.reshape([batch_size*zu_size])
    else:
        brand_text=brand_text_[step*batch_size*zu_size:(step+1)*batch_size*zu_size]
        brand_pic=brand_pic_[step*batch_size*zu_size:(step+1)*batch_size*zu_size]
        in_text=in_text_[step*batch_size*zu_size:(step+1)*batch_size*zu_size]
        in_pic=in_pic_[step*batch_size*zu_size:(step+1)*batch_size*zu_size]
        label_=labels_[step*batch_size*zu_size:(step+1)*batch_size*zu_size]
        label_ = label_.reshape([batch_size*zu_size])
        #print('label:',label_)
    return brand_text,brand_pic,in_text,in_pic,label_

def fill_feed_dict_train(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train, brand_text_pl,brand_pic_pl, in_text_pl,in_pic_pl, labels_pl,step,keep_prob):
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  brand_text_feed,brand_pic_feed,in_text_feed,in_pic_feed,labels_feed = get_batch(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train,step)
  feed_dict = {
      brands_text: brand_text_feed,
      brands_pic:brand_pic_feed,
      influencers_text: in_text_feed,
      influencers_pic:in_pic_feed,
      labels: labels_feed,
      keep_prob:0.5,
  }
  return feed_dict


def fill_feed_dict_test(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train, brand_text_pl,brand_pic_pl, in_text_pl,in_pic_pl, labels_pl,step,keep_prob):
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  brand_text_feed,brand_pic_feed,in_text_feed,in_pic_feed,labels_feed = get_batch(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train,step)
  feed_dict = {
      brands_text: brand_text_feed,
      brands_pic:brand_pic_feed,
      influencers_text: in_text_feed,
      influencers_pic:in_pic_feed,
      labels: labels_feed,
      keep_prob:1,
  }
  return feed_dict

def metrics(l_user,l_in,l_ist,l_score):#MedR,recall@10,recall@50
    all_positive=790
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
    ExcelFile1=xlrd.open_workbook('G:\\aaa\\test_set_auc_v3.xlsx')
    sheet1=ExcelFile1.sheet_by_index(0)
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
        a=sheet1.cell(i,0).value.encode('utf-8').decode('utf-8-sig')
        b=sheet1.cell(i,1).value.encode('utf-8').decode('utf-8-sig')
        c=sheet1.cell(i,3).value.encode('utf-8').decode('utf-8-sig')
        d=sheet1.cell(i,4).value.encode('utf-8').decode('utf-8-sig')
        e=sheet1.cell(i,5).value
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
graph1 = tf.Graph()
with graph1.as_default():
    
    keep_prob = tf.placeholder(tf.float32)
    
    influencers_text,influencers_pic=influencer_vectors_inputs()#pic_size=25088,text_size=300
    brands_text,brands_pic=brand_vector_inputs()
    labels=label_vector_inputs()
    brands_pic_=tf.reshape(brands_pic,[batch_size*zu_size,7,7,512])
    influencers_pic_=tf.reshape(influencers_pic,[batch_size*zu_size,7,7,512])
    normal_brands_pic=tf.nn.local_response_normalization(brands_pic_,2,0.1,1,1)
    normal_influencers_pic=tf.nn.local_response_normalization(influencers_pic_,2,0.1,1,1)
    
    normal_brands_pic=tf.reshape(normal_brands_pic,[batch_size*zu_size,25088])
    normal_influencers_pic=tf.reshape(normal_influencers_pic,[batch_size*zu_size,25088])
    
    
    
    
    w_brand_text1=tf.Variable(tf.random_normal([text_size,text_layer1_size],stddev=0.1),name="text_weights1_brands")
    dropout1=tf.nn.dropout(w_brand_text1,keep_prob)
    b_brand_text1=tf.Variable(tf.zeros([text_layer1_size],name="bias_brands_text_1"))#batch size*300  *  300*text_layer1_size  =50*text_layer1_size +1*text_layer1_size
    
    w_in_text1=tf.Variable(tf.random_normal([text_size,text_layer1_size],stddev=0.1),name="text_weights1_influencers")  
    dropout2=tf.nn.dropout(w_in_text1,keep_prob)
    b_in_text1=tf.Variable(tf.zeros([text_layer1_size],name="bias_influencers_text_1"))#batch size *300   *   300*text_layer1_size = batch size *text_layer1_size +1*text_layer1_size
    
    brand_text_embed_v1=tf.nn.leaky_relu(tf.matmul(brands_text,dropout1)+b_brand_text1,0.01)
    in_text_embed_v1=tf.nn.leaky_relu(tf.matmul(influencers_text,dropout2)+b_in_text1,0.01)
    
    w_brand_text4=tf.Variable(tf.random_normal([text_layer1_size,text_layer4_size],stddev=0.1),name="text_weights4_brands")
    w_in_text4=tf.Variable(tf.random_normal([text_layer1_size,text_layer4_size],stddev=0.1),name="text_weights4_influencers")
    
    brand_text_embed_v4=tf.matmul(brand_text_embed_v1,w_brand_text4)
    in_text_embed_v4=tf.matmul(in_text_embed_v1,w_in_text4)
    
    
    
    
    w_brand_pic1=tf.Variable(tf.random_normal([pic_size,pic_layer1_size],stddev=0.1),name="pic_weights1_brands")
    dropout3=tf.nn.dropout(w_brand_pic1,keep_prob)
    b_brand_pic1=tf.Variable(tf.zeros([pic_layer1_size],name="bias_brands_pic_1"))#batch size*300  *  300*pic_layer1_size  =50*pic_layer1_size +1*pic_layer1_size
    
    w_in_pic1=tf.Variable(tf.random_normal([pic_size,pic_layer1_size],stddev=0.1),name="pic_weights1_influencers")
    dropout4=tf.nn.dropout(w_in_pic1,keep_prob)
    b_in_pic1=tf.Variable(tf.zeros([pic_layer1_size],name="bias_influencers_pic_1"))#batch size *300   *   300*pic_layer1_size = batch size *pic_layer1_size +1*pic_layer1_size
    
    brand_pic_embed_v1=tf.nn.leaky_relu(tf.matmul(normal_brands_pic,dropout3)+b_brand_pic1,0.01)
    in_pic_embed_v1=tf.nn.leaky_relu(tf.matmul(normal_influencers_pic, dropout4)+b_in_pic1,0.01)
    
    w_brand_pic4=tf.Variable(tf.random_normal([pic_layer1_size,pic_layer4_size],stddev=0.1),name="pic_weights4_brands")
    w_in_pic4=tf.Variable(tf.random_normal([pic_layer1_size,pic_layer4_size],stddev=0.1),name="pic_weights4_influencers")
    
    brand_pic_embed_v4=tf.matmul(brand_pic_embed_v1,w_brand_pic4)
    in_pic_embed_v4=tf.matmul(in_pic_embed_v1,w_in_pic4)
    #pic与text做fusion,得到embed_v3



    
    brand_embed=tf.multiply(brand_text_embed_v4,brand_pic_embed_v4)
    in_embed=tf.multiply(in_text_embed_v4,in_pic_embed_v4)
    
    
    product_1=tf.multiply(brand_embed,in_embed)
    x=tf.reduce_mean(product_1,axis=1)
    y=tf.reshape(x,[batch_size,zu_size])
    y_1=tf.nn.softmax(y)
    y_2=tf.reshape(y_1,[batch_size*zu_size])
    y_2=y_2+1e-8
    cross_entropy2=-tf.reduce_mean(tf.reduce_sum(labels*tf.log(y_2)))
    
    
    
    
    lr_base=0.001
    lr_decay=0.99
    lr_step=400
    global_steps=tf.Variable(0,trainable=False)
    lr=tf.train.exponential_decay(lr_base,global_steps,lr_step,lr_decay,staircase=True)
    train_op=tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy2)



    
def main():
    Epoch_=30
    part_=56
    Step_=0
    Epoch=0
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    config.gpu_options.visible_device_list = "0"
    #先引入dataset
    
    with tf.Session(graph=graph1) as sess:
    
        saver = tf.train.Saver(max_to_keep=30)
    
    
        init=tf.global_variables_initializer()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.all_model_checkpoint_paths:
            path_=''
            
            for path in ckpt.all_model_checkpoint_paths:
                path_=path
            
            print(path_)
            saver.restore(sess, path_)
        else:
            init=tf.global_variables_initializer()
            sess.run(init)
        


        for j in range(0,Epoch_):
            #print('epoch:',j)
            for i in range(0,part_):
                part=i
                #print('part:',i)
                print('loading...train')
                path1='F:\\dataset_k_4_train\\brand_text_test_'+str(i)+'.npy'
                brand_text_train=np.load(path1)
                print('loading...train')
                in_text_train=np.load('F:\\dataset_k_4_train\\in_text_test_'+str(i)+'.npy')
                print('loading...train')
                brand_pic_train=np.load('F:\\dataset_k_4_train\\brand_pic_test_'+str(i)+'.npy')
                print('loading...train')
                in_pic_train=np.load('F:\\dataset_k_4_train\\in_pic_test_'+str(i)+'.npy')
                print('loading...train')
                labels_train=np.load('F:\\dataset_k_4_train\\label_test_'+str(i)+'.npy')
                if(len(brand_text_train)%(zu_size*batch_size)==0):
                    Step_=len(brand_text_train)/(zu_size*batch_size)
                else:
                    Step_=int(len(brand_text_train)/(zu_size*batch_size))
                Step_=int(Step_)
                
                
                print('Epoch %d, Part %d'%(Epoch,part))
                mean_loss=0
                for step in xrange(Step_):
                    start_time = time.time()
                    feed_dict = fill_feed_dict_train(brand_text_train,brand_pic_train,in_text_train,in_pic_train,labels_train,brands_text,brands_pic,influencers_text,influencers_pic,labels,step,keep_prob)
                    
                    _brand_pic,_brand_embed, _in_embed_v4, _labels, _x,_y=sess.run([y_2,brand_text_embed_v4, x, labels,w_brand_pic1, w_brand_text1],feed_dict=feed_dict)
                    
                    _, loss_value = sess.run([train_op, cross_entropy2],feed_dict=feed_dict)
                    mean_loss+=loss_value
                    
                    duration = time.time() - start_time
                    
                    if (step % 20 == 0 and step!=0):
                            
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, mean_loss/step, duration))
                        
                    globalstep=part+(56*Epoch)+1
                    if ((globalstep%56==0 or (globalstep%56==28 and Epoch>15)) and (step==Step_-1) and globalstep!=0):
                        
                        checkpoint_file = os.path.join(model_path, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=globalstep)
                        
                if ((part+1)==56):
                    print('-----test-----')
                    ExcelFile1=xlrd.open_workbook('G:\\aaa\\test_set.xlsx')
                    sheet1=ExcelFile1.sheet_by_index(0)
                    l_user=[]
                    l_in=[]
                    l_ist=[]
                    l_score=[]
                    index=0
                    for k in range(0,2):
                        print('loading...test')
                        path1='F:\\dataset_k_4_test\\brand_text_test_'+str(k)+'.npy'
                        brand_text_test=np.load(path1)
                        print('loading...test')
                        in_text_test=np.load('F:\\dataset_k_4_test\\in_text_test_'+str(k)+'.npy')
                        print('loading...test')
                        brand_pic_test=np.load('F:\\dataset_k_4_test\\brand_pic_test_'+str(k)+'.npy')
                        print('loading...test')
                        in_pic_test=np.load('F:\\dataset_k_4_test\\in_pic_test_'+str(k)+'.npy')
                        print('loading...test')
                        labels_test=np.load('F:\\dataset_k_4_test\\label_test_'+str(k)+'.npy')
                        if(len(brand_text_test)%(zu_size*batch_size)==0):
                            tStep_=len(brand_text_test)/(zu_size*batch_size)
                        else:
                            tStep_=int(len(brand_text_test)/(zu_size*batch_size))
                        tStep_=int(tStep_)
                        test_mean_loss=0.0
                        
                        for t in xrange(tStep_):
                            
                            feed_dict = fill_feed_dict_test(brand_text_test,brand_pic_test,in_text_test,in_pic_test,labels_test,brands_text,brands_pic,influencers_text,influencers_pic,labels,t,keep_prob)
                            test_labels, test_x,test_loss=sess.run([labels, x,cross_entropy2],feed_dict=feed_dict)
                            test_mean_loss+=test_loss
                            if(t % 10 == 0 and t!=0):
                                print('Step %d: loss = %.2f ' % (t, test_mean_loss/t))
                            for xx in test_x:
                                
                                user=sheet1.cell(index,0).value.encode('utf-8').decode('utf-8-sig')
                                influencer=sheet1.cell(index,1).value.encode('utf-8').decode('utf-8-sig')
                                ist=sheet1.cell(index,2).value
                                l_user.append(user)
                                l_in.append(influencer)
                                l_ist.append(ist)
                                l_score.append(xx)
                                index+=1
                metrics(l_user,l_in,l_ist,l_score)
                auc(l_user,l_in,l_ist,l_score)
                    
            Epoch+=1
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    