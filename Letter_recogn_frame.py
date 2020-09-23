from Neural_network import *
##Make node inputs
def create_input(image):
    result=[]
    for i in range(len(image)):
        for j in range(len(image)):
            result.append(image[i][j])
    return np.asmatrix(result)

###make node outputs
def create_output(target):
    result=[]
    for i in range(10):
        if i==target:
            result.append(1)
        else:
            result.append(0)
    return result

##deterimine output, given by a matrix 1 is the highest
def output(result):
    dummy=0
    for i in range(result.size):
        if result.item(i)>=dummy:
            dummy=result.item(i)
            output=i
    if dummy==0:      
         return 'NONE'
    return output

def prediction(W,inn,bias,digits):
    inn_1=create_input(digits.images[inn])
    z=Neural_Network_nodes(inn_1,W,bias)
    a=sigma(z)
    return output(a[-1]), digits.target[inn]

def training_digits(W,d,bias,alpha,digits):
    inn=create_input(digits.images[d])
    out=create_output(digits.target[d])
    W1=train_NN(inn,out,W,bias,alpha)
    return W1

###do multiple training examples of a set between start and stop
def training_NN_digits(W,bias,start,stop,alpha,digits):
    for i in range(stop-start):
        W,bias=training_digits(W,i,bias,alpha,digits)
    return W,bias

####Check how accurate the model is for multiple values
def Accuracy(W,bias,start,stop,digits,debug=False):
    num=0
    length=stop-start
    for i in range(start,stop):
        k=prediction(W,i,bias,digits)
        if k[0]==k[1]:
            num+=1
        elif debug==True:
            print(k[0],k[1])
            #plt.gray()
            #plt.matshow(digits.images[i])
            #plt.show()
            #input('press enter to continue')
    print('Number of correct', num, '\n Number Total', length)

    return num/length
            
