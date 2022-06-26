#!/usr/bin/env python
# coding: utf-8

# In[1]:
#pip install -U sentence-transformers


# In[2]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint


# In[3]:


model= SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# In[4]:


print("Para que el modelo funcione correctamente tienes que respondes a las preguntas con frases y evitar los monosílabos")


# In[5]:


print("¿Cómo de triste te sientes hoy?")
respuestas1=['No me siento triste',
            'Me siento triste gran parte del tiempo',
            'Me siento triste todo el tiempo',
            'Me siento tan triste o soy tan infeliz que no puedo soportarlo']
preg1= input()
embeddings1=model.encode(respuestas1)
input1=model.encode(preg1)


#variable sumatorio de puntuaciones

sumatorio=0


# In[6]:


largo1= len(embeddings1)
print(largo1)


# In[7]:


len(embeddings1[0])


# In[8]:


r11=cosine_similarity(embeddings1[0].reshape(1,-1), input1.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt11=0
pprint(r11)


# In[9]:


r12=cosine_similarity(embeddings1[1].reshape(1,-1), input1.reshape(1,-1))
#Puntuacion asignada es 01ptos.
punt12=1
pprint(r12)


# In[10]:


r13=cosine_similarity(embeddings1[2].reshape(1,-1), input1.reshape(1,-1))
#Puntuacion asignada son 2 ptos.
punt13=2
pprint(r13)


# In[11]:


r14=cosine_similarity(embeddings1[3].reshape(1,-1), input1.reshape(1,-1))
#Puntuacion asignada es 3 ptos.
punt14=3
pprint(r14)


# In[12]:


lista_respuestas1=[r11, r12, r13, r14]

print(lista_respuestas1)


# In[13]:


sumatorio=0


# In[14]:



aux=r11
i=0
punt1=0
for i in range (largo1) :
    if (lista_respuestas1[i] > aux):
           
           aux= lista_respuestas1[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)
        


# In[15]:


print("¿Cómo ves tu futuro?")
respuestas2=['No estoy desalentado respecto del mi futuro.',
            'Me siento más desalentado respecto de mi futuro que lo que solía estarlo. ',
            'No espero que las cosas funcionen para mi.',
            'Siento que no hay esperanza para mi futuro y que sólo puede empeorar.']
preg2= input()
embeddings2=model.encode(respuestas2)
input2=model.encode(preg2)


# In[16]:


largo2= len(embeddings2)
print(largo2)


# In[17]:


r21=cosine_similarity(embeddings2[0].reshape(1,-1), input2.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt21=0
pprint(r21)


# In[18]:


r22=cosine_similarity(embeddings2[1].reshape(1,-1), input2.reshape(1,-1))
#Puntuacion asignada es 01ptos.
punt22=1
pprint(r22)


# In[19]:


r23=cosine_similarity(embeddings2[2].reshape(1,-1), input2.reshape(1,-1))
#Puntuacion asignada es 01ptos.
punt23=2
pprint(r23)


# In[20]:


r24=cosine_similarity(embeddings2[3].reshape(1,-1), input2.reshape(1,-1))
#Puntuacion asignada es 01ptos.
punt24=3
pprint(r24)


# In[21]:


lista_respuestas2=[r21, r22, r23, r24]

print(lista_respuestas2)


# In[22]:


aux=r21
i=0
punt1=0
for i in range (largo2) :
    if (lista_respuestas2[i] > aux):
          
           aux= lista_respuestas2[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[23]:


print("¿Cómo te sientes respecto a tus fracasos?")
respuestas3=['No me siento como un fracasado.',
            'He fracasado más de lo que hubiera debido.',
            'Cuando miro hacia atrás, veo muchos fracasos.',
            'Siento que como persona soy un fracaso total.']
preg3= input()
embeddings3=model.encode(respuestas3)
input3=model.encode(preg3)


# In[26]:


largo3= len(embeddings3)
print(largo3)


# In[27]:


r31=cosine_similarity(embeddings3[0].reshape(1,-1), input3.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt31=0
pprint(r31)


# In[28]:


r32=cosine_similarity(embeddings3[1].reshape(1,-1), input3.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt31=1
pprint(r32)


# In[29]:


r33=cosine_similarity(embeddings3[2].reshape(1,-1), input3.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt31=2
pprint(r33)


# In[30]:


r34=cosine_similarity(embeddings3[3].reshape(1,-1), input3.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt31=3
pprint(r34)


# In[31]:


lista_respuestas3=[r31, r32, r33, r34]

print(lista_respuestas3)


# In[32]:


aux=r31
i=0
punt1=0
for i in range (largo3) :
    if (lista_respuestas3[i] > aux):
           
           aux= lista_respuestas3[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[33]:


print("¿Sigues disfrutando de las cosas como antes?")
respuestas4=['Obtengo tanto placer como siempre por las cosas de las que disfruto.',
            'No disfruto tanto de las cosas como solía hacerlo.',
            'Obtengo muy poco placer de las cosas que solía disfrutar. ',
            'No puedo obtener ningún placer de las cosas de las que solía disfrutar.']
preg4= input()
embeddings4=model.encode(respuestas4)
input4=model.encode(preg4)


# In[34]:


largo4= len(embeddings4)
print(largo4)


# In[35]:


r41=cosine_similarity(embeddings4[0].reshape(1,-1), input4.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt41=0
pprint(r41)


# In[36]:


r42=cosine_similarity(embeddings4[1].reshape(1,-1), input4.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt42=0
pprint(r42)


# In[37]:


r43=cosine_similarity(embeddings4[2].reshape(1,-1), input4.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt43=0
pprint(r43)


# In[38]:


r44=cosine_similarity(embeddings4[3].reshape(1,-1), input4.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt44=0
pprint(r44)


# In[39]:


lista_respuestas4=[r41, r42, r43, r44]

print(lista_respuestas4)


# In[40]:


aux=r41
i=0
punt1=0
for i in range (largo4) :
    if (lista_respuestas4[i] > aux):
           
           aux= lista_respuestas4[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[41]:


print("¿Tienes sentimientos de culpa?")
respuestas5=['No me siento particularmente culpable.',
            'Me siento culpable respecto de varias cosas que he hecho o que debería haber hecho.',
            'Me siento bastante culpable la mayor parte del tiempo.',
            'Me siento culpable todo el tiempo.']
preg5= input()
embeddings5=model.encode(respuestas5)
input5=model.encode(preg5)


# In[42]:


largo5= len(embeddings5)
print(largo5)


# In[44]:


r51=cosine_similarity(embeddings5[0].reshape(1,-1), input5.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt51=0
pprint(r51)


# In[45]:


r52=cosine_similarity(embeddings5[1].reshape(1,-1), input5.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt52=0
pprint(r52)


# In[46]:


r53=cosine_similarity(embeddings5[2].reshape(1,-1), input5.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt53=0
pprint(r53)


# In[47]:


r54=cosine_similarity(embeddings5[3].reshape(1,-1), input5.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt54=0
pprint(r54)


# In[48]:


lista_respuestas5=[r51, r52, r53, r54]

print(lista_respuestas5)


# In[49]:


aux=r51
i=0
punt1=0
for i in range (largo5) :
    if (lista_respuestas5[i] > aux):
           
           aux= lista_respuestas5[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[50]:


print("¿Sientes que deberías o estás siendo castigado?")
respuestas6=['No siento que este siendo castigado',
            'Siento que tal vez pueda ser castigado. ',
            'Espero ser castigado.',
            'Siento que estoy siendo castigado.']
preg6= input()
embeddings6=model.encode(respuestas6)
input6=model.encode(preg6)


# In[51]:


largo6= len(embeddings6)
print(largo6)


# In[52]:


r61=cosine_similarity(embeddings6[0].reshape(1,-1), input6.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt61=0
pprint(r61)


# In[53]:


r62=cosine_similarity(embeddings6[1].reshape(1,-1), input6.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt62=0
pprint(r62)


# In[54]:


r63=cosine_similarity(embeddings6[2].reshape(1,-1), input6.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt63=0
pprint(r63)


# In[55]:


r64=cosine_similarity(embeddings6[3].reshape(1,-1), input6.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt64=0
pprint(r64)


# In[56]:


lista_respuestas6=[r61, r62, r63, r64]

print(lista_respuestas6)


# In[57]:


aux=r61
i=0
punt1=0
for i in range (largo6) :
    if (lista_respuestas6[i] > aux):
           
           aux= lista_respuestas6[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[58]:


print("¿Qué confianza tienes en ti mismo?")
respuestas7=['Siento acerca de mi lo mismo que siempre.',
            'He perdido la confianza en mí mismo. ',
            'Estoy decepcionado conmigo mismo.',
            'No me gusto a mí mismo.']
preg7= input()
embeddings7=model.encode(respuestas7)
input7=model.encode(preg7)


# In[59]:


largo7= len(embeddings7)
print(largo7)


# In[60]:


r71=cosine_similarity(embeddings7[0].reshape(1,-1), input7.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt71=0
pprint(r71)


# In[61]:


r72=cosine_similarity(embeddings7[1].reshape(1,-1), input7.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt72=0
pprint(r72)


# In[62]:


r73=cosine_similarity(embeddings7[2].reshape(1,-1), input7.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt71=0
pprint(r73)


# In[63]:


r74=cosine_similarity(embeddings7[3].reshape(1,-1), input7.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt71=0
pprint(r74)


# In[64]:


lista_respuestas7=[r71, r72, r73, r74]

print(lista_respuestas7)


# In[65]:


aux=r71
i=0
punt1=0
for i in range (largo7) :
    if (lista_respuestas7[i] > aux):
           
           aux= lista_respuestas7[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[66]:


print("¿Te criticas a ti mismo con facilidad?")
respuestas8=['No me critico ni me culpo más de lo habitual.',
            'Estoy más crítico conmigo mismo de lo que solía estarlo.',
            'Me critico a mí mismo por todos mis errores.',
            'Me culpo a mí mismo por todo lo malo que sucede.']
preg8= input()
embeddings8=model.encode(respuestas8)
input8=model.encode(preg8)


# In[67]:


largo8= len(embeddings8)
print(largo8)


# In[68]:


r81=cosine_similarity(embeddings8[0].reshape(1,-1), input8.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt81=0
pprint(r81)


# In[69]:


r82=cosine_similarity(embeddings8[1].reshape(1,-1), input8.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt82=0
pprint(r82)


# In[70]:


r83=cosine_similarity(embeddings8[2].reshape(1,-1), input8.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt81=0
pprint(r83)


# In[71]:


r84=cosine_similarity(embeddings8[3].reshape(1,-1), input8.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt81=0
pprint(r84)


# In[72]:


lista_respuestas8=[r81, r82, r83, r84]

print(lista_respuestas8)


# In[73]:


aux=r81
i=0
punt1=0
for i in range (largo8) :
    if (lista_respuestas8[i] > aux):
           
           aux= lista_respuestas8[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[74]:


print("¿Tienes pensamientos o deseos suicidas?")
respuestas9=['No tengo ningún pensamiento de matarme.',
            'He tenido pensamientos de matarme, pero no lo haría',
            'Querría matarme ',
            'Me mataría si tuviera la oportunidad de hacerlo.']
preg9= input()
embeddings9=model.encode(respuestas9)
input9=model.encode(preg9)


# In[75]:


largo9= len(embeddings9)
print(largo9)


# In[76]:


r91=cosine_similarity(embeddings9[0].reshape(1,-1), input9.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt91=0
pprint(r91)


# In[77]:


r92=cosine_similarity(embeddings9[1].reshape(1,-1), input9.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt91=0
pprint(r92)


# In[78]:


r93=cosine_similarity(embeddings9[2].reshape(1,-1), input9.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt91=0
pprint(r93)


# In[79]:


r94=cosine_similarity(embeddings9[3].reshape(1,-1), input9.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt91=0
pprint(r94)


# In[81]:


lista_respuestas9=[r91, r92, r93, r94]

print(lista_respuestas9)


# In[82]:


aux=r91
i=0
punt1=0
for i in range (largo9) :
    if (lista_respuestas9[i] > aux):
           
           aux= lista_respuestas9[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[83]:


print("¿Con qué frecuencia lloras?")
respuestas10=['No lloro más de lo que solía hacerlo.',
            'Lloro más de lo que solía hacerlo',
            'Lloro por cualquier pequeñez.',
            'Siento ganas de llorar pero no puedo.']
preg10= input()
embeddings10=model.encode(respuestas10)
input10=model.encode(preg10)


# In[84]:


largo10= len(embeddings10)
print(largo10)


# In[85]:


r101=cosine_similarity(embeddings10[0].reshape(1,-1), input10.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r101)


# In[87]:


r102=cosine_similarity(embeddings10[1].reshape(1,-1), input10.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r102)


# In[88]:


r103=cosine_similarity(embeddings10[2].reshape(1,-1), input10.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r103)


# In[89]:


r104=cosine_similarity(embeddings10[3].reshape(1,-1), input10.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r104)


# In[90]:


lista_respuestas10=[r101, r102, r103, r104]

print(lista_respuestas10)


# In[91]:


aux=r101
i=0
punt1=0
for i in range (largo10) :
    if (lista_respuestas10[i] > aux):
           
           aux= lista_respuestas10[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[92]:


print("¿Estás agitado o tenso normalmente?")
respuestas11=['No estoy más inquieto o tenso que lo habitual.',
            'Me siento más inquieto o tenso que lo habitual.',
            'Estoy tan inquieto o agitado que me es difícil quedarme quieto ',
            'Estoy tan inquieto o agitado que tengo que estar siempre en movimiento o haciendo algo.']
preg11= input()
embeddings11=model.encode(respuestas11)
input11=model.encode(preg11)


# In[93]:


largo11= len(embeddings11)
print(largo11)


# In[94]:


r111=cosine_similarity(embeddings11[0].reshape(1,-1), input11.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r111)


# In[95]:


r112=cosine_similarity(embeddings11[1].reshape(1,-1), input11.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r112)


# In[96]:


r113=cosine_similarity(embeddings11[2].reshape(1,-1), input11.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r113)


# In[97]:


r114=cosine_similarity(embeddings11[3].reshape(1,-1), input11.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt104=0
pprint(r114)


# In[98]:


lista_respuestas11=[r111, r112, r113, r114]

print(lista_respuestas11)


# In[99]:


aux=r111
i=0
punt1=0
for i in range (largo11) :
    if (lista_respuestas11[i] > aux):
           
           aux= lista_respuestas11[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[100]:


print("¿Sientes pérdida de interés en actividades o personas?")
respuestas12=['No he perdido el interés en otras actividades o personas.',
            'Estoy menos interesado que antes en otras personas o cosas.',
            'He perdido casi todo el interés en otras personas o cosas.',
            'Me es difícil interesarme por algo.']
preg12= input()
embeddings12=model.encode(respuestas12)
input12=model.encode(preg12)


# In[101]:


largo12= len(embeddings12)
print(largo12)


# In[102]:


r121=cosine_similarity(embeddings12[0].reshape(1,-1), input12.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r121)


# In[103]:


r122=cosine_similarity(embeddings12[1].reshape(1,-1), input12.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r122)


# In[104]:


r123=cosine_similarity(embeddings12[2].reshape(1,-1), input12.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r123)


# In[105]:


r124=cosine_similarity(embeddings12[3].reshape(1,-1), input12.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r124)


# In[106]:


lista_respuestas12=[r121, r122, r123, r124]

print(lista_respuestas12)


# In[107]:


aux=r121
i=0
punt1=0
for i in range (largo12) :
    if (lista_respuestas12[i] > aux):
           
           aux= lista_respuestas12[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[108]:


print("¿Sueles tener problemas a la hora de tomar decisiones?")
respuestas13=['Tomo mis propias decisiones tan bien como siempre.',
            'Me resulta más difícil que de costumbre tomar decisiones.',
            'Encuentro mucha más dificultad que antes para tomar decisiones.',
            'Tengo problemas para tomar cualquier decisión.']
preg13= input()
embeddings13=model.encode(respuestas13)
input13=model.encode(preg13)


# In[109]:


largo13= len(embeddings13)
print(largo13)


# In[110]:


r131=cosine_similarity(embeddings13[0].reshape(1,-1), input13.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r131)


# In[111]:


r132=cosine_similarity(embeddings13[1].reshape(1,-1), input13.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r132)


# In[112]:


r133=cosine_similarity(embeddings13[2].reshape(1,-1), input13.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r133)


# In[113]:


r134=cosine_similarity(embeddings13[3].reshape(1,-1), input13.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r134)


# In[114]:


lista_respuestas13=[r131, r132, r133, r134]

print(lista_respuestas13)


# In[115]:


aux=r131
i=0
punt1=0
for i in range (largo13) :
    if (lista_respuestas13[i] > aux):
           
           aux= lista_respuestas13[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[116]:


print("¿Cómo de valioso te sientes?")
respuestas14=['No siento que yo no sea valioso',
            'No me considero a mi mismo tan valioso y útil como solía considerarme',
            'Me siento menos valioso cuando me comparo con otros',
            'Siento que no valgo nada.']
preg14= input()
embeddings14=model.encode(respuestas14)
input14=model.encode(preg14)


# In[117]:


largo14= len(embeddings14)
print(largo14)


# In[118]:


r141=cosine_similarity(embeddings14[0].reshape(1,-1), input14.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r141)


# In[119]:


r142=cosine_similarity(embeddings14[1].reshape(1,-1), input14.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r142)


# In[120]:


r143=cosine_similarity(embeddings14[2].reshape(1,-1), input14.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r143)


# In[121]:


r144=cosine_similarity(embeddings14[3].reshape(1,-1), input14.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r144)


# In[122]:


lista_respuestas14=[r141, r142, r143, r144]

print(lista_respuestas14)


# In[123]:


aux=r141
i=0
punt1=0
for i in range (largo14) :
    if (lista_respuestas14[i] > aux):
           
           aux= lista_respuestas14[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[124]:


print("¿Te notas con menos energía?")
respuestas15=['Tengo tanta energía como siempre.',
            'Tengo menos energía que la que solía tener. ',
            'No tengo suficiente energía para hacer demasiado. ',
            'No tengo energía suficiente para hacer nada. ']
preg15= input()
embeddings15=model.encode(respuestas15)
input15=model.encode(preg15)


# In[125]:


largo15= len(embeddings15)
print(largo15)


# In[126]:


r151=cosine_similarity(embeddings15[0].reshape(1,-1), input15.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r151)


# In[127]:


r152=cosine_similarity(embeddings15[1].reshape(1,-1), input15.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r152)


# In[128]:


r153=cosine_similarity(embeddings15[2].reshape(1,-1), input15.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r153)


# In[129]:


r154=cosine_similarity(embeddings15[3].reshape(1,-1), input15.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt101=0
pprint(r154)


# In[130]:


lista_respuestas15=[r151, r152, r153, r154]

print(lista_respuestas15)


# In[131]:


aux=r151
i=0
punt1=0
for i in range (largo15) :
    if (lista_respuestas15[i] > aux):
           
           aux= lista_respuestas15[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[132]:


print("¿Presentas cambios en los hábitos del sueño?")
respuestas16=['No he experimentado ningún cambio en mis hábitos de sueño.',
            'Duermo un poco más que lo habitual.',
            'Duermo un poco menos que lo habitual.',
            'Duermo mucho más que lo habitual.',
            'Duermo mucho menos que lo habitual',
             'Duermo la mayor parte del día',
             'Me despierto 1-2 horas más temprano y no puedo volver a dormirme']
preg16= input()
embeddings16=model.encode(respuestas16)
input16=model.encode(preg16)


# In[133]:


largo16= len(embeddings16)
print(largo16)


# In[134]:


r161=cosine_similarity(embeddings16[0].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r161)


# In[135]:


r162=cosine_similarity(embeddings16[1].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r162)


# In[136]:


r163=cosine_similarity(embeddings16[2].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r163)


# In[137]:


r164=cosine_similarity(embeddings16[3].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r164)


# In[138]:


r165=cosine_similarity(embeddings16[4].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r165)


# In[139]:


r166=cosine_similarity(embeddings16[5].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r166)


# In[140]:


r167=cosine_similarity(embeddings16[6].reshape(1,-1), input16.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r167)


# In[141]:


lista_respuestas16=[r161, r162, r163, r164, r165, r166, r167]

print(lista_respuestas16)


# In[149]:


aux=r161
i=0
punt1=0
for i in range (largo16) :
    if (lista_respuestas16[i] > aux):
           
           aux= lista_respuestas16[i]
          # punt1=i
            
if (aux == r161):
    punt1 = 0
elif (aux == r162):
    punt1 =1
elif (aux == r163):
    punt1 =1
elif (aux == r164):
    punt1 = 2
elif (aux == r165):
    punt1 = 2
elif (aux == r166):
    punt1 = 3
elif (aux == r167):
    punt1 =3
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[150]:


print("¿Cómo de irritable estás últimamente?")
respuestas17=['No estoy tan irritable que lo habitual.',
            'Estoy más irritable que lo habitual.',
            'Estoy mucho más irritable que lo habitual.',
            'Estoy irritable todo el tiempo.']
preg17= input()
embeddings17=model.encode(respuestas17)
input17=model.encode(preg17)


# In[151]:


largo17= len(embeddings17)
print(largo17)


# In[152]:


r171=cosine_similarity(embeddings17[0].reshape(1,-1), input17.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r171)


# In[153]:


r172=cosine_similarity(embeddings17[1].reshape(1,-1), input17.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r172)


# In[160]:


r173=cosine_similarity(embeddings17[2].reshape(1,-1), input17.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r173)


# In[161]:


r174=cosine_similarity(embeddings17[3].reshape(1,-1), input17.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r174)


# In[162]:


lista_respuestas17=[r171, r172, r173, r174]

print(lista_respuestas17)


# In[163]:


aux=r171
i=0
punt1=0
for i in range (largo17) :
    if (lista_respuestas17[i] > aux):
           
           aux= lista_respuestas17[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[164]:


print("¿Presentas cambios en el apetito?")
respuestas18=['No he experimentado ningún cambio en mi apetito. ',
            'Mi apetito es un poco menor que lo habitual.',
            'Mi apetito es un poco mayor que lo habitual.',
            'Mi apetito es mucho menor que antes.',
             'Mi apetito es mucho mayor que lo habitual',
             'No tengo apetito en absoluto.',
             'Quiero comer todo el día.']
preg18= input()
embeddings18=model.encode(respuestas18)
input18=model.encode(preg18)


# In[165]:


largo18= len(embeddings18)
print(largo18)


# In[166]:


r181=cosine_similarity(embeddings18[0].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r181)


# In[167]:


r182=cosine_similarity(embeddings18[1].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r182)


# In[168]:


r183=cosine_similarity(embeddings18[2].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r183)


# In[169]:


r184=cosine_similarity(embeddings18[3].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r184)


# In[170]:


r185=cosine_similarity(embeddings18[4].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r185)


# In[171]:


r186=cosine_similarity(embeddings18[5].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r186)


# In[172]:


r187=cosine_similarity(embeddings18[6].reshape(1,-1), input18.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r187)


# In[173]:


lista_respuestas18=[r181, r182, r183, r184, r185, r186, r187]

print(lista_respuestas18)


# In[174]:


aux=r181
i=0
punt1=0
for i in range (largo18) :
    if (lista_respuestas18[i] > aux):
           
           aux= lista_respuestas18[i]
          # punt1=i
            
if (aux == r181):
    punt1 = 0
elif (aux == r182):
    punt1 =1
elif (aux == r183):
    punt1 =1
elif (aux == r184):
    punt1 = 2
elif (aux == r185):
    punt1 = 2
elif (aux == r186):
    punt1 = 3
elif (aux == r187):
    punt1 =3
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[175]:


print("Describe tu facilidad/dificultad para concentrarte ultimamente")
respuestas19=['Puedo concentrarme tan bien como siempre.',
            'No puedo concentrarme tan bien como habitualmente',
            'Me es difícil mantener la mente en algo por mucho tiempo.',
            'Encuentro que no puedo concentrarme en nada.']
preg19= input()
embeddings19=model.encode(respuestas19)
input19=model.encode(preg19)


# In[176]:


largo19= len(embeddings19)
print(largo19)


# In[177]:


r191=cosine_similarity(embeddings19[0].reshape(1,-1), input19.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r191)


# In[178]:


r192=cosine_similarity(embeddings19[1].reshape(1,-1), input19.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r192)


# In[179]:


r193=cosine_similarity(embeddings19[2].reshape(1,-1), input19.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r193)


# In[180]:


r194=cosine_similarity(embeddings19[3].reshape(1,-1), input19.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r194)


# In[181]:


lista_respuestas19=[r191, r192, r193, r194]

print(lista_respuestas19)


# In[182]:


aux=r191
i=0
punt1=0
for i in range (largo19) :
    if (lista_respuestas19[i] > aux):
           
           aux= lista_respuestas19[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[183]:


print("¿Qué nivel de fatiga presentas ultimamente?")
respuestas20=['No estoy más cansado o fatigado que lo habitual.',
            'Me fatigo o me canso más fácilmente que lo habitual.',
            'Estoy demasiado fatigado o cansado para hacer muchas de las cosas que solía hacer.',
            'Estoy demasiado fatigado o cansado para hacer la mayoría de las cosas que solía hacer']
preg20= input()
embeddings20=model.encode(respuestas20)
input20=model.encode(preg20)


# In[184]:


largo20= len(embeddings20)
print(largo20)


# In[185]:


r201=cosine_similarity(embeddings20[0].reshape(1,-1), input20.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r201)


# In[186]:


r202=cosine_similarity(embeddings20[1].reshape(1,-1), input20.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r202)


# In[187]:


r203=cosine_similarity(embeddings20[2].reshape(1,-1), input20.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r203)


# In[188]:


r204=cosine_similarity(embeddings20[3].reshape(1,-1), input20.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r204)


# In[189]:


lista_respuestas20=[r201, r202, r203, r204]

print(lista_respuestas20)


# In[190]:


aux=r201
i=0
punt1=0
for i in range (largo20) :
    if (lista_respuestas20[i] > aux):
           
           aux= lista_respuestas20[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[194]:


print("¿Sientes una pérdida de interés en el sexo ultimamente?")
respuestas21=['No he notado ningún cambio reciente en mi interés por el sexo.',
            'Estoy menos interesado en el sexo de lo que solía estarlo. ',
            'Estoy mucho menos interesado en el sexo. ',
            'He perdido completamente el interés en el sexo.']
preg21= input()
embeddings21=model.encode(respuestas21)
input21=model.encode(preg21)


# In[195]:


largo21= len(embeddings21)
print(largo21)


# In[196]:


r211=cosine_similarity(embeddings21[0].reshape(1,-1), input21.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r211)


# In[198]:


r212=cosine_similarity(embeddings21[1].reshape(1,-1), input21.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r212)


# In[199]:


r213=cosine_similarity(embeddings21[2].reshape(1,-1), input21.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r213)


# In[200]:


r214=cosine_similarity(embeddings21[3].reshape(1,-1), input21.reshape(1,-1))
#Puntuacion asignada es 0 ptos.
punt161=0
pprint(r214)


# In[201]:


lista_respuestas21=[r211, r212, r213, r214]

print(lista_respuestas21)


# In[202]:


aux=r211
i=0
punt1=0
for i in range (largo21) :
    if (lista_respuestas21[i] > aux):
           
           aux= lista_respuestas21[i]
           punt1=i
          
#variable sumatorio de las puntuaciones
sumatorio += punt1
   
print(aux)

print(sumatorio)


# In[203]:


if(sumatorio< 14):
    pprint("Usted padece depresión mínima")
elif( 13< sumatorio <20 ):
    pprint("Usted padece depresión leve")
elif (19< sumatorio <29):
    pprint("Usted padece depresión moderada")
else:
    pprint("Usted padece depresión grave")
    


# In[ ]:




