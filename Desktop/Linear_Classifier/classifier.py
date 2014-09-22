import random
from numpy import *
import numpy
class dataitem:
        classes = 0
        attributes = []
        def __init__(self, class1, point):
                self.classes = class1
                self.attributes = point
	
class dataset:
    #file1=[]
    values=[]

    #Number of classes in each dataset
    numClasses = 0

    
    def __init__(self,item):
        self.values=item
        
    def read(self,filename):
        file1=[]
        file=open(filename, 'r')
        for eachline in file.readlines():
            file1.append(eachline)
        random.shuffle(file1)
        list_of_points=[]
        
        for entry in file1:
            a=entry.split(" ")
            class1=[]
            point=[]
            class1.append(int(a[0]))
            for i in a[1:]:
                point.append(float(i))
        #print point
            list_of_points.append(point)
            temp=dataitem(class1,point)
            self.values.append(temp)
            #print len(class1)
        file.close()
        return True

    def write(self,filename):
        file=open(filename, 'w')
        list_of_points=[]
        for entry in file1:
            a=entry.split(" ")
            class1=[]
            point=[]
            class1.append(int(a[0]))
            for i in range(1,n+1):
                point.append(float(a[i]))
        #print point
            list_of_points.append(point)
            temp=dataitem(class1,point)
            self.values.append(temp)
        file.close()
        return False




def splitDataset(complete,folds=5,seed=0):
    dataPartitions = []
    min = 0
    jump = len(complete.values)/folds
            
    for i in range (0,folds):

        tdataSet=dataset(complete.values[min:jump])
        min=min+jump
        jump=min+jump

        dataPartitions.append(tdataSet)

    return dataPartitions

def mergeDataset(tomerge,numDatasets,indecesToMerge):
        
   
    for i in tomerge:
        tomer=dataset(tomer.values)+i.values
    return tomer
            

class LinearClassifier:
        
    #options={0:SinglePerceptron,
     #        1:BatchPerceptron,
      #       2:SingleRelaxation,
       #      3:BatchRelaxation,
        #     4:MSEPseudoInverse,
         #    5:MSELMSProcedure,}
    



    def matrix(self,training,classes1,classes2):
            dim=len(training.values[0].attributes)
            n=len(training.values)
            y = zeros((1,dim+1))
            x=ones((1,dim+1))
            err=0
            threshhold=0.1
            for dataitem in training.values:
                    #print len(training.values)
                    if dataitem.classes[0] in classes1:
                            temp_vector=dataitem.attributes
                            temp_vector.insert(0,1)
                            z=numpy.matrix(temp_vector)
                            y=vstack((y,z))
                    elif dataitem.classes[0] in classes2:
                            temp_vector=dataitem.attributes
                            temp_vector.insert(0,1)
                            z=numpy.matrix(temp_vector)
                            z=-z
                            y=vstack((y,z))
                             
            #print y.shape[1]    
            y=y[1:,:]
            #print y.shape[0]
            return y
           

    def SinglePerceptron(self,y_values):
            wt = numpy.matrix(ones((y_values.shape[1],1)))

          
            k=0
            while k<1000:
                            
                    for index in range(y_values.shape[0]):
                        mat=dot(y_values[index],wt)
                        
                        if mat>0:
                                continue
                        else:
                                wt=wt+y_values[index].transpose()
                    k+=1
            count=0
            for index in range(y_values.shape[0]):
                            
                            mat=dot(y_values[index],wt)
                            if mat<0:
                                    count+=1
            #print count
            err=(1.0*count)/len(y_values)

            return wt,err


    def BatchPerceptron(self,y_values,eta,theta):
            wt = numpy.matrix(ones((y_values.shape[1],1)))
            k=1
            count=0
            delta=ones((y_values.shape[1],1))+theta
            while (absolute(delta)<theta).all()==False:
                    count+=1
                    delta=delta*0
                    tempcount=0
                    for index in range(y_values.shape[0]):
                        mat=dot(y_values[index],wt)
                        if mat>0:
                                continue
                        else:
                                tempcount+=1
                                delta+=y_values[index].transpose()
                    delta=delta*eta(k)
                    wt+=delta
                    k+=1

            count=0
            for index in range(y_values.shape[0]):
                    
                            
                            mat=dot(y_values[index],wt)
                            if mat<0:
                                    count+=1
            #print count
            err=(1.0*count)/len(y_values)

            return wt,err
            
    def SingleRelaxation(self,y_values,eta,b):
            wt=numpy.matrix(ones((y_values.shape[1],1)))
            k=1
            count=0
            while k<1000:
                    for index in range (y_values.shape[0]):
                            mat=dot(y_values[index],wt)
                            if mat>0:
                                    continue
                            else:
                                    delta=y_values[index].transpose()*(((1.0*b)-dot(y_values[index],wt))/dot(y_values[index],y_values[index].transpose()))
                                    delta=delta*eta(k)
                                    wt+=delta
                    k=k+1
            for index in range(y_values.shape[0]):
                    
                            
                            mat=dot(y_values[index],wt)
                            if mat<0:
                                    count+=1
            #print count
            err=(1.0*count)/len(y_values)
                    
            return wt,err
    def BatchRelaxation(self,y_values,eta,b):
            wt=numpy.matrix(ones((y_values.shape[1],1)))
            k=1
            count=0
            
            delta=ones((y_values.shape[1],1))
            while k<1000:
                    tempcount=0
                    delta=delta*0
                    for index in range (y_values.shape[0]):
                            mat=dot(y_values[index],wt)
                            if mat>b:

                                    continue
                            else:
                                    tempcount+=1
                                    delta+=y_values[index].transpose()*(((1.0*b)-dot(y_values[index],wt))/dot(y_values[index],y_values[index].transpose()))
                    delta=delta*((1.0/k))
                    wt+=delta
                    k+=1
                    count+=1
                    
                   
            #print 'count',count
            count=0
            for index in range(y_values.shape[0]):
                    
                            
                            mat=dot(y_values[index],wt)
                            if mat<b:

                                    count+=1
            #print count
            err=(1.0*count)/len(y_values)

            return wt,err
    def MSEPseudoInverse(self,y_values):
            wt=numpy.matrix(ones((y_values.shape[1],1)))
            b=ones((y_values.shape[0],1))
            wt=dot(linalg.pinv(y_values),b)
            print "returning just weight"
            return wt
    def MSELMSProcedure(self,y_values,eta,theta):
            wt = numpy.matrix(ones((y_values.shape[1],1)))
            #print wt
            k=0
            count=0
            b=ones((y_values.shape[0],1))
            delta=ones((y_values.shape[1],1))+theta
            while (absolute(delta)<theta).all()==False and count<=150:
                    count+=1
                    for index in range(y_values.shape[0]):
                            mat=dot(y_values[index],wt)
                            if mat>0:
                                    continue
                            else:
                                    delta=(b[k]-dot(y_values[index],wt))*(y_values[index])
                                    delta=delta*eta(k+1)
                                    wt+=delta.transpose()
                                    k+=1
            #print count
            print "returning just weight"                        
            return wt
    def inverse_number(self,num):
            return 1.0/num
    def tenth_root_inverse_number(self,num):
            return 1.0/(num**0.1)

    def getClassSize(self, tDataSet):
            classNameSet = set()
            for i in range(0,len(tDataSet.values)):
                    tempSet = tDataSet.values[i].classes
                    classNameSet.update(tempSet)
            #print classNameSet 
            self.numClasses = len(classNameSet)
            print "no. of classes"
            print self.numClasses

    
    def one_vs_rest(self):
            classNames = set()
            classPairs = []

            for i in range(self.numClasses):
                    tempSet = set([i])
                    classNames.update(tempSet)

            for i in range(len(classNames)):
                    setone = set([i])
                    settwo = classNames - setone
                    classPairs.append([setone,settwo])
            print "class pairs"        

            print classPairs
    def one_vs_one(self):
            classPairs=[]
            for i in range(self.numClasses):
                    setone=set([i])
                    for j in range(i+1,self.numClasses):
                            settwo=set([j])
                            classPairs.append([setone,settwo])
            print "class pairs"
            print classPairs                

datairis=dataset([])
datairis.read('C:\\Users\\lokesh\\Desktop\\iris_new.txt')
data_set_iris_parts=splitDataset(datairis,1)
classifier=LinearClassifier()
classifier.getClassSize(datairis)
classifier.one_vs_rest()
#classifier.one_vs_one()
 # one vs rest is running. For Running classifiers for other classes replace the classes no. in the line below i.e for iris classes are (0,1,2)
y_matrix=classifier.matrix(data_set_iris_parts[0], set([0]), set([1,2])) # classified (0 class with set(1,2)) all classes can be classified by jst replacing the classes which you want to classify
#print y_matrix
# for running each algo uncomment that algo call and run it. 1 algo should be uncomment at a time
a,err = classifier.SinglePerceptron(y_matrix)
#a,err = classifier.BatchPerceptron(y_matrix, classifier.inverse_number, 0.001)
#a,err = classifier.SingleRelaxation(y_matrix, classifier.tenth_root_inverse_number, 0.00001)
#a,err = classifier.BatchRelaxation(y_matrix, classifier.inverse_number,0.00001)
#a= classifier.MSEPseudoInverse(y_matrix)
#a = classifier.MSELMSProcedure(y_matrix, classifier.inverse_number, 0.1)
print "weight"
print a
print "error in percent"
print err

