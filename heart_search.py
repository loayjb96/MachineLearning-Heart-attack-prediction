
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import decimal
import itertools
heart= pd.read_csv('heart.csv')


# In[2]:


# our goal is to try understand the data we got lets start 
# here we can see the info of each attribute  in the heart dataframe  as you can see it is not null 
#let us rename the colm we  have to be more understandable 
heart.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',
       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']

arr=[]
fig=plt.figure(figsize=(10, 10))

# for i in heart:
#     ax=sns.countplot(x=i,data=heart)
#     fig=plt.figure(figsize=(10, 10))


# In[3]:



# this is the data plot of all the attrr counter of each value in attr 

sns.heatmap(heart[heart.columns[:13]].corr(),annot=True,cmap='RdYlGn')
fig=plt.gcf()
fig.set_size_inches(20,18)
plt.suptitle('Correlation Matrix')
plt.savefig('new_plot/correlationofdata')


# In[4]:



#### lets know split the data we have in order to start the learning procsses 
heart.head()
X_data = heart.drop(columns=['Target'], axis=1)
Y = heart['Target']
#normalize the data
Y = ((Y - np.min(Y))/ (np.max(Y) - np.min(Y))).values
X = ((X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))).values
# we want here to seprate the data for the learning procsses testdata=30% train 70 %
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3)

# let know use two type of classfication algortimm i will first use logstic regression 
logisticRegr=LogisticRegression()
logisticRegr.fit(x_train, y_train)
# wtemp={0:1.5,1:1}
# logisticRegrl1=LogisticRegression(C=0.6,tol=1,class_weight=wtemp)
# logisticRegrl1.fit(x_train, y_train)

pred = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)


# predl1 = logisticRegrl1.predict(x_test)
# scorel1 = logisticRegrl1.score(x_test, y_test)
# print('__________________________________________________________')
# print(" score of abdoslute score is ",scorel1)
# print('__________________________________________________________')

error_LR=1-score
print("score: ",score)
cmlogstic=confusion_matrix(y_test,pred)
# def result_score_of_model(model,ytest):    
def plot_heat(cm,title): 
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells

# labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['not sick', 'sick']); ax.yaxis.set_ticklabels(['not sick', 'sick']);
    plt.savefig('new_plot/heatmap'+title)
    
    plt.show()
    
print("Error of logstic:  ",error_LR)
def plot_roc(x_test,model,title):
    
    probs = model.predict_proba(x_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def plot_roc_svm(x_test,model,title):
    classes = model.predict(X)
    p = np.array(model.decision_function(x_test)) # decision is a voting function

    probs = model.predict(x_test)
#     preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
#     plt.savfig('new')
    plt.show()

    
#     p = np.array(clf.decision_function(x_test)) # decision is a voting function
#     prob = np.exp(p)/np.sum(np.exp(p),axis=1) # softmax after the voting
    
      
           
#     plot_roc(x_test,logisticRegr,'logsticreg roc')
#     plot_heat(cmlogstic," logstic")
##
##############################################3
def get_report(y_test,pred):
    Report = [ (accuracy_score(y_test, pred),f1_score(y_test, pred, average='weighted')
                ,recall_score(y_test, pred), precision_score(y_test, pred))]
    dfObj = pd.DataFrame(Report, columns=['Accuracy', 'F1 score','Recall','Precision'], index=['1'])
    print('\n clasification report:\n', dfObj)   
    print ('\n clasification report for pair:\n', classification_report(y_test,pred))
get_report(y_test,pred)


# In[5]:



# usin svm learning algotrim 
types=['rbf','linear']
for i in types:
    model=svm.SVC(kernel=i)
    model.fit(x_train,y_train)
#     predsvm=mo.predict(x_test)
    svm_prediction=model.predict(x_test)
    cmsvm=confusion_matrix(y_test,svm_prediction)
    
    plot_heat(cmsvm,i+' svm ')
#     print('Accuracy for SVM kernel=',i,'is',metrics.accuracy_score(svm_prediction,y_test))
#     print("report of ",i)
#     get_report(y_test,svm_prediction)
    if(i=='linear'):
        plot_roc_svm(x_test,model,'svm')
# plot_roc_svm(x_test,model,'svm')
print('***********************************')
# wwtemp={0:1.4,1:1}

# modelff=svm.SVC(kernel='linear',C=10,tol=0.1,class_weight=wwtemp)
# modelff.fit(x_train,y_train)
# y_pred_sepal = modelff.predict(x_test)

# www=accuracy_score(y_test,y_pred_sepal)
# print('**********************')

# print(www)


# In[6]:



# know i want to play with the logstic regression paramter wwe will start with c 
# which is nverse of regularization strengt meaning we will force the theata value to be in spsfic range 
# let us start using the parmater we already have the train test above 


# let us start using the parmater we already have the train test above 


import pandas as pd
def get_max_min_scor_of(sepal_acc_table):
    maxem=sepal_acc_table[sepal_acc_table['Accuracy']==sepal_acc_table['Accuracy'].max()]
    minm=sepal_acc_table[sepal_acc_table['Accuracy']==sepal_acc_table['Accuracy'].min()]
    min_value=np.asarray(minm)
    max_value=np.asarray(maxem)
#     print(min_value)
    return max_value[0],min_value[0]
# def plot_two_grph(list1,list2,title,x_title,axis_range):
#     plt.plot(axis_range,list1)
# #     plt.plot(list1,marker='o',color='red' ,label="test_plot")

# #     plt.plot(list2,marker='o',color='black',label ="train_plot")
#     plt.axis=axis_range
#     plt.title(title)
#     plt.xlabel(x_title)
#     plt.ylabel('Accuracy')

#     plt.legend()
#     plt.show()
def plot_two_grph(list1,list2,title,x_title):
    plt.plot(list1,marker='o',color='red' ,label="test_plot")
    plt.plot(list2,marker='o',color='black',label ="trainplot")
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('Accuracy')

    plt.legend()
    plt.savefig('new_plot/two_graph'+title)
    plt.ticklabel_format(useOffset=False)

    plt.show()

def plot_two_dict(dict1,dict2,title,x_title):
    if(x_title=='attribute'):
        
        plt.plot(list(dict1.keys()), list(dict1.values()),label="logstic ",marker='o')
        plt.plot(list(dict2.keys()), list(dict2.values()),label="svm",marker='o')
    else:     
        plt.plot(list(dict1.keys()), list(dict1.values()),label="train_plot",marker='o')
#     plt.label("train")
        plt.plot(list(dict2.keys()), list(dict2.values()),label="test_plot",marker='o')
#     plt.label("test")

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('Accuracy')

    plt.legend()
    plt.savefig('new_plot/plot_2dict'+title)
    plt.show()

    
def plot_x_y(x,y,title,taple):
    
    max_g_tol,min_g_tol=get_max_min_scor_of(sepal_acc_table)
    fig, ax = plt.subplots()
    ax.plot(x,y,marker='o', color='g')
    # naming the x axis 
    plt.xlabel(title) 
    # naming the y axis 
    plt.ylabel('Accuracy') 
    # giving a title to my graph 
    plt.title(title)
    text1='MaxValue at ('+ str(max_g_tol[0])+' ,'+ str(max_g_tol[1])+')'
    text2='MinValue at ('+ str(min_g_tol[0])+' ,'+ str(min_g_tol[1])+')'
    ax.annotate(text1, xy=max_g_tol, xytext=(6.4,0.83),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));
    ax.annotate(text2, xy=min_g_tol, xytext=(6.4,0.80),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));
#     ax.ticklabel_format(useOffset=True)
    plt.ticklabel_format(useOffset=False)

    plt.savefig('new_plot/max_min'+title)

    plt.show()

j=0
def itr_throug_param(Logstic_parmter,name_of_par_col,paramlist,table,title,score_data_x,score_data_y,model_type):
    sepal_acc_table=table
    sepal_acc_table[name_of_par_col] = paramlist
    j=0
#     print(sepal_acc_table)
    for i in paramlist:
      # Apply logistic regression model to training data
        if(Logstic_parmter=='C'):
            if(model_type=="logstic"):
                lr = LogisticRegression(C= i,random_state = 0)
            if(model_type=='svm'):
#                 print('svm')
                lr=svm.SVC(C=i)

                
        if(Logstic_parmter=='tol'):
            if(model_type=="logstic"):
                lr = LogisticRegression(tol= i,random_state = 0)
            if(model_type=='svm'):
                lr=svm.SVC(tol=i)

        if(Logstic_parmter=='class_weight'):
            temp={0:i,1:1}
            lr = LogisticRegression(class_weight = temp,random_state = 0)
            if(model_type=="logstic"):
                lr = LogisticRegression(class_weight= temp,random_state = 0)
            if(model_type=='svm'):
#                 print('svm')
                lr=svm.SVC(class_weight= temp)
        

        lr.fit(x_train,y_train)
        y_pred_sepal = lr.predict(score_data_x)
    # Saving accuracy score in table
#         accuarc=accuracy_score(y_test,y_pred_sepal)
        sepal_acc_table.iloc[j,1] = accuracy_score(score_data_y,y_pred_sepal)
        j += 1
        x=sepal_acc_table[name_of_par_col]
        y=sepal_acc_table['Accuracy']
        copy_taple=sepal_acc_table.copy()

    plot_x_y(x,y,title,sepal_acc_table)
    
    ret=copy_taple.set_index(name_of_par_col).T.to_dict('list')
# c_table_train_dict=c_table_train.set_index('C_parameter').T.to_dict('list')
    return ret

C_param_range = [0.00001,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2,10,20]
sepal_acc_table = pd.DataFrame(columns = ['C_parameter','Accuracy'])
c_table_test=itr_throug_param('C','C_parameter',C_param_range,sepal_acc_table,'c paramter change',x_test,y_test,'logstic')
c_table_train=itr_throug_param('C','C_parameter',C_param_range,sepal_acc_table,'c paramter change',x_train,y_train,'logstic')

# c_table_test_dict=c_table_test.set_index('C_parameter').T.to_dict('list')
# c_table_train_dict=c_table_train.set_index('C_parameter').T.to_dict('list')
plot_two_dict(c_table_train,c_table_test,'changing c ',' C_parameter ') # this wil plot the accuarcy changin in both train and test 
    
    
# plot_two_grph(c_table_test_dict,c_table_train_dict,'title','x_title',C_param_range)
print(" _________________________________________svm _____________________________-")

# svm_sepal_acc_table = pd.DataFrame(columns = ['C_parameter','Accuracy'])
svm_c_table_test=itr_throug_param('C','C_parameter',C_param_range,sepal_acc_table,'c paramter change',x_test,y_test,'svm')
svm_c_table_train=itr_throug_param('C','C_parameter',C_param_range,sepal_acc_table,'c paramter change',x_train,y_train,'svm')
plot_two_dict(svm_c_table_train,svm_c_table_test,'changing c ',' C_parameter ') # this wil plot the accuarcy changin in both train and test 
    


# In[7]:


### i want to check the tolerance value changes 
tol_par = [0.0001,0.00001,0.00005,0.000003,0.0001,0.001,0.01,0.1,1,2,5,10,20,25]
sepal_acc_table = pd.DataFrame(columns = ['tol_par','Accuracy'])
sepal_acc_table['tol_par'] = tol_par
# tol_table=itr_throug_param('tol','tol_par',tol_par,sepal_acc_table,'tol paramter change')
tol_table_train=itr_throug_param('tol','tol_par',tol_par,sepal_acc_table,'tol paramter change',x_train,y_train,'logstic')
tol_table_test=itr_throug_param('tol','tol_par',tol_par,sepal_acc_table,'c paramter change',x_test,y_test,'logstic')
plot_two_dict(tol_table_train,tol_table_test,'changing tol  ',' tol_parameter ') # this wil plot the accuarcy changin in both train and test 
        
print("__________________________________svm_________________________________")
svm_tol_table_train=itr_throug_param('tol','tol_par',tol_par,sepal_acc_table,'tol paramter change',x_train,y_train,'svm')
svm_tol_table_test=itr_throug_param('tol','tol_par',tol_par,sepal_acc_table,'c paramter change',x_test,y_test,'svm')
plot_two_dict(svm_tol_table_train,svm_tol_table_test,'changing tol  ',' tol_parameter ') # this wil plot the accuarcy changin in both train and test 

# plot_x_y(x,y," graph for accuracy of changing tol par ",sepal_acc_table)

# print(sepal_acc_table)


# In[8]:


# class_weight="balanced"
## so lets check the target arrange we have 
def drange(x, y, jump):
      while x < y:
            yield float(x)
            x += jump
sepal_acc_table = pd.DataFrame(columns = ['Class_w','Accuracy'])         
class_w1= drange(0.5, 4,0.05)
class_w1=[x for x in class_w1]
sepal_acc_table['Class_w'] = class_w1
# heart['Target'].value_counts()
class_w2= drange(1, 2,0.1)
index=0
# w_table=itr_throug_param('class_weight','Class_w',class_w1,sepal_acc_table,'class wight paramter change',x_data,y_data)
w_table_train=itr_throug_param('class_weight','Class_w',class_w1,sepal_acc_table,'class_weight paramter change',x_train,y_train,'logstic')
w_table_test=itr_throug_param('class_weight','Class_w',class_w1,sepal_acc_table,'class_weight paramter change',x_test,y_test,'logstic')
plot_two_dict(w_table_train,w_table_test,'changing class_weight  ',' class_weight ') # this wil plot the accuarcy changin in both train and test 
print("____________________________svm_w_____________________________________")    

svm_w_table_train=itr_throug_param('class_weight','Class_w',class_w1,sepal_acc_table,'class_weight paramter change',x_train,y_train,'svm')
svm_w_table_test=itr_throug_param('class_weight','Class_w',class_w1,sepal_acc_table,'class_weight paramter change',x_test,y_test,'svm')
plot_two_dict(svm_w_table_train,svm_w_table_test,'changing class_weight  ',' class_weight ') # this wil plot the accuarcy changin in both train and t
# x=w_table['Class_w']
# y=w_table['Accuracy']
# plot_x_y(x,y,'class wight paramter change',sepal_acc_table)
# print(w_table)


# In[9]:


### so know we want to do for svc 

   # Apply logistic regression model to training data
# max_w,min_w=get_max_min_scor_of(w_table_train)
import operator
max_w=max(w_table_train.items(), key=operator.itemgetter(1))[0]
max_c=max(c_table_train.items(), key=operator.itemgetter(1))[0]
max_tol=max(tol_table_train.items(), key=operator.itemgetter(1))[0]

print("____",max_w)
print("max class_weight is ",max_w)
# max_c,min_c=get_max_min_scor_of(c_table_train)
print("max C_parmeter is  ",max_c)
# max_tol,min_tol=get_max_min_scor_of(tol_table_train)
print("max Tol_parmeter is  ",max_tol)

temp={0:max_w,1:1}

lrn = LogisticRegression(tol=max_tol,C=max_c,class_weight = temp)
lrn.fit(x_train,y_train)
score=lrn.score(x_test,y_test)
# score=lrn.score(x_train,y_train)

print('the logstic regresstion score  when taking all the par that gived us max is: ')
print(score)
print("________________________________svm____________________________________________")


svm_max_w=max(svm_w_table_train.items(), key=operator.itemgetter(1))[0]
svm_max_c=max(svm_c_table_train.items(), key=operator.itemgetter(1))[0]
svm_max_tol=max(svm_tol_table_train.items(), key=operator.itemgetter(1))[0]

print("____",svm_max_w)
print("max class_weight is ",svm_max_w)
# max_c,min_c=get_max_min_scor_of(c_table_train)
print("max C_parmeter is  ",svm_max_c)
# max_tol,min_tol=get_max_min_scor_of(tol_table_train)
print("max Tol_parmeter is  ",svm_max_tol)

temp={0:svm_max_w,1:1}

svm_lrn = svm.SVC(tol=svm_max_tol,C=svm_max_c,class_weight = temp)
svm_lrn.fit(x_train,y_train)
svm_score=svm_lrn.score(x_test,y_test)
# score=lrn.score(x_train,y_train)
print('the logstic regresstion score  when taking all the par that gived us max is: ')
print(svm_score)


# In[10]:


### know i want to use all the paramter of class wight and c paramte and tol to check which is the best 
# C_param_range,tol_par,class_w1
### note we handeld the overfiting by chosing c 0.3 which did give us better 
def get_score_with_defer_par(C_param_range,tol_par,class_w1,model):
    score_list=[]
    dictlist={}
    for c in C_param_range:
        for tol in tol_par:
            for w in class_w1:
                temp_w={0:w,1:1}  
                if(model=='logstic'):
                    lr = LogisticRegression(C=c,tol=tol,class_weight=temp_w,random_state = 0,penalty='l2')
                if(model=='svm'):
                    lr = svm.SVC(C=c,tol=tol,class_weight=temp_w,random_state = 0)
#                     print('svm')

                lr.fit(x_train,y_train)
                score=lr.score(x_test,y_test)
#                 print(score)
                dict1={"c_par":c,"w":w,"tol":tol}
                score_dict={score:dict1}
                dictlist[score] = dict1
                score_list.append(score)
                
    return score_list,dictlist
score,dictlist=get_score_with_defer_par(C_param_range,tol_par,class_w1,'logstic')
key_list = list(dictlist.keys()) 
val_list = list(dictlist.values()) 
values=(key_list)
max_index= key_list.index(max(key_list))
temp_dict=val_list[max_index]
cm=temp_dict['c_par']
wm=temp_dict['w']
tol=temp_dict['tol']
print("C_parmters: ",cm,"Tol_parmter: ",tol,"ClassWeight_parmter: ",wm)
print("the best score we got: ",max(score))



print("______________________________________ know we check for svm ___________________________")





# In[11]:


svm_t_score,svm_t_dictlist=get_score_with_defer_par(C_param_range,tol_par,class_w1,'svm')
svm_key_list = list(svm_t_dictlist.keys()) 
svm_val_list = list(svm_t_dictlist.values()) 
values=(svm_key_list)
# print(" values ",values)
svm_max_index= svm_key_list.index(max(svm_key_list))
svm_temp_dict=svm_val_list[svm_max_index]
svm_cm=svm_temp_dict['c_par']
svm_wm=svm_temp_dict['w']
svm_tol=svm_temp_dict['tol']
print("C_parmters: ",svm_cm,"Tol_parmter: ",svm_tol,"ClassWeight_parmter: ",svm_wm)
print("the best score we got: ",max(svm_t_score))


# In[12]:


# print(svm_key_list)


# In[13]:


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
temp_w={0:max_w,1:1}
model=LogisticRegression()
attr_col = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',
       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia']
used_bool=[]
def get_best_attr_with_score(model,x_test,y_test,attr_col,k):
#     X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
#     estimator = SVR(kernel="linear")
    selector = RFE(model, k, step=1)
#     print("________________1")
    selector = selector.fit(x_train, y_train)
#     selector.support_
#     print("________________2")

    selector.ranking_
    our_selected_attr=selector.get_support()
    list_attr_se=[]
    score=selector.score(x_test,y_test)
    for i ,j in zip(our_selected_attr,attr_col):
        if(i==True):
            list_attr_se.append(j)
    return score,list_attr_se,our_selected_attr

def get_score_list_attr(model,x_test,y_test,attr_col):
    scores_of_kattr=[]
    dict_attr={}
    dict_attr_score={}
    for i in range(0,len(attr_col)):
        score,attr_list,Used_attr=get_best_attr_with_score(model,x_test,y_test,attr_col,i+1)
        scores_of_kattr.append(score)
        dict_attr[i] = attr_list
        dict_attr_score[i]=score
        dfObjClass = pd.DataFrame(Used_attr[i], columns=['ChestPain', 'MajorVessels','Slope','Oldpeak','Gender','Thalassemia','MaxHeartRateAchivied','Age','ExerciseIndusedAngina','RestingECG','RestingBloodPressure','Cholestrol','FastingBloodSugar'], index=[i+1])
       
    return scores_of_kattr,dict_attr,dict_attr_score

attr_score,attr_list,ll=get_score_list_attr(model,x_test,y_test,attr_col)

for i in attr_list:
    print(attr_list[i],"-> (",attr_score[i],")")
plt.plot(attr_score,marker='o')
plt.title('logstic most important attr ')
plt.savefig('new_plot/logstic_attr_check')

plt.show()
clf = svm.SVC(kernel='linear')

print(' svm ____________________')
svm_temp_w={0:svm_max_w,1:1}

svm_model = svm.SVC(C=svm_max_c,tol=svm_max_tol,class_weight=temp_w,random_state = 0)

svm_attr_score,svm_attr_list,llsvm=get_score_list_attr(clf,x_test,y_test,attr_col)
for i in svm_attr_list:
    print(svm_attr_list[i],"-> (",svm_attr_score[i],")")
plt.plot(svm_attr_score,marker='o')
plt.title('svm most important attr ')

plt.savefig('new_plot/svm_attr_check')
plt.show()


plot_two_dict(ll,llsvm,' most important attribute','attribute')

# print(svm_attr_score)


# In[14]:


from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, Y, cv=10)
print(scores)


# In[15]:


from sklearn.model_selection import KFold
from sklearn.svm import SVR

X = np.asarray((heart.drop(columns=['Target'], axis=1)))
y = np.asarray(heart['Target'])
kf = KFold(shuffle=True, n_splits=10)
kf.get_n_splits(X)

# print(kf)
# def kfold_on_model(X,Y,model):
train_svm_kfold_score=[]
test_svm_kfold_score=[]
test_logstic_kfold_score=[]

train_logstic_kfold_score=[]
for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ### lets try svm model accuaracy 
    temp_w={0:max_w,1:1}
    lr = LogisticRegression(C=max_c,tol=max_tol,class_weight=temp_w,random_state = 0,penalty='l2')
    svm_temp_w={0:svm_max_w,1:1}
#     svml = SVR(kernel='rbf')

    svml = svm.SVC(C=svm_max_c,tol=svm_max_tol,class_weight=svm_temp_w,random_state = 0)
    lr.fit(X_train,y_train)
    score_test=lr.score(X_test,y_test)
    score_train=lr.score(X_train,y_train)
    train_logstic_kfold_score.append(score_train)
    test_logstic_kfold_score.append(score_train)
    svml.fit(X_train,y_train)
    svm_score_test=svml.score(X_test,y_test)
    svm_score_train=svml.score(X_train,y_train)
    train_svm_kfold_score.append(svm_score_train)
    test_svm_kfold_score.append(svm_score_test)

    print('-----------------------------------------------------------')
    print("svm      test score ",svm_score_test," s scovm train re ",svm_score_train)
    print("logstic  test score ",score_test ," logstic train score ",score_train)
    print("___________________________________________________________________")

    
    




    
    

