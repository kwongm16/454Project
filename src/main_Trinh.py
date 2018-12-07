'''
Created on Nov 24, 2018

@author: micha and trinh
'''
#Importing packages used
import openpyxl
import numpy as np

#Importing data from "Line_Data.xlsx" excel document
workbook = openpyxl.load_workbook('454 Data Project.xlsx') 

MVAbase = 100

#Making each sheet of data (Line & Load data) more accessable
lineData = workbook['LineData'] 
busData = workbook['BusData'] 

#Initializing arrays to store Pload,Qload,V,Bus Type
#Where we assume bus 1 is the "Slack Bus" 
Pload = [] 
Qload = [] 
Pgen = [] 
V = [] 
busType = [] 

#Putting BusData into arrays
for row_index in range(1, (busData.max_row)): 
    Pload.append(busData.cell(row=row_index + 1, column=2).value/MVAbase)
    Qload.append(busData.cell(row=row_index + 1, column=3).value/MVAbase)
    Pgen.append(busData.cell(row=row_index + 1, column=4).value/MVAbase)
    V.append(busData.cell(row=row_index + 1, column=5).value)
    busType.append(busData.cell(row=row_index + 1, column=6).value)
    
# print('Pload:',Pload)
# print('Qload:',Qload) 
# print('Pgen:', Pgen)
# print('V:', V)
# print('busType:', busType)

# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#Initializing Y-Admittitance matrix where the size is based on the 
#number of buses in the grid
numBuses = 12
yBus = np.zeros(shape=(numBuses, numBuses), dtype=complex)

for row_index in range (0, (lineData.max_row - 1)): 
    
    sendIndex = lineData.cell(row=row_index + 2, column=1).value
    receiveIndex = lineData.cell(row_index + 2, column=2).value
    
    rTotal = lineData.cell(row_index + 2, column=3).value
    xTotal = lineData.cell(row_index + 2, column=4).value
    bTotal = lineData.cell(row_index + 2, column=5).value
    seriesImpedance = complex(rTotal, xTotal)
    
    #Populating off-diagonal terms
    yBus[sendIndex - 1, receiveIndex - 1] = -1*(1 / seriesImpedance)
    yBus[receiveIndex - 1, sendIndex - 1] = -1*(1 / seriesImpedance)

    #Populating diagondal terms
    yBus[sendIndex - 1, sendIndex - 1] += (1 / seriesImpedance) + (1j * bTotal / 2)
    yBus[receiveIndex - 1, receiveIndex - 1] += (1 / seriesImpedance) + (1j * bTotal / 2)

#print(yBus)
#a = np.asarray(yBus)
#np.savetxt("foo.csv", a, delimiter=",")

# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#Convergence requirements
Pconv = 0.1
Qconv = 0.1

#Number of PQ and PV buses
numPV = 0
numPQ = 0
#Injections have all known P injections first followed by known Q injections

#Count how many PQ and PV buses in system
for idx in range(0,len(busType)):
    if busType[idx] == "PV":
        numPV += 1

    elif busType[idx] == "PQ":
        numPQ += 1


#fill unknown V's with flat start guesses
for idx in range(0, numBuses):
    if busType[idx] =="PQ":
        V[idx] = 1

#intialize a theta array with knowns and flat start guesses
theta = np.zeros(numBuses)


#Function to identify indexes of PQ buses
def pqIdx():
    pqIdx = []
    for idx in range(0, numBuses):
        if busType[idx] =="PQ":
            pqIdx.append(idx+1)
    return pqIdx

pqIdx = pqIdx()

#Function for P computed
def Pcomp(j):
    Pcomp = 0
    for k in range(0, numBuses):
        Pcomp += V[j] * V[k] * (np.real(yBus[j][k])*np.cos(theta[j]-theta[k]) + np.imag(yBus[j][k])*np.sin(theta[j]-theta[k]))
    return Pcomp

#Function for Q computed
def Qcomp(i):
    Qcomp = 0
    for k in range(0, numBuses):
        Qcomp += V[i] * V[k] * (np.real(yBus[i][k])*np.sin(theta[i]-theta[k]) - np.imag(yBus[i][k])*np.cos(theta[i]-theta[k]))
    return Qcomp

#Function for P mismatch
def deltaP():
    deltaP = np.zeros(numBuses-1)
    for j in range(0, numBuses - 1):
        deltaP[j] = (Pgen[j+1]-Pload[j+1]) - Pcomp(j+1)
    return deltaP

#Function for Q mismatch
def deltaQ():
    deltaQ = np.zeros(numPQ)
    for j in range(0, numPQ):
        deltaQ[j] = (-1*Qload[pqIdx[j]-1]) - Qcomp(pqIdx[j]-1)
    return deltaQ


#Define Jacobian submatrices
def J11():
    J11 = np.zeros(shape = (numBuses-1, numBuses-1))

    for j in range(0,numBuses-1):
        for k in range(0,numBuses-1):
            if (k) == (j):
                J11[j][k] = -1*Qcomp(j+1) -1*V[j+1]**2*np.imag(yBus[j+1][j+1])
            else:
                J11[j][k] = V[j+1] * V[k+1] *(np.real(yBus[j+1][k+1])*np.sin(theta[j+1]-theta[k+1]) - np.imag(yBus[j+1][k+1])*np.cos(theta[j+1]-theta[k+1]))
    return J11


def J12():
    J12 = np.zeros(shape = (numBuses-1, numPQ))

    for j in range(0,numBuses-1):
        for k in range (0,numPQ):
            kprime = pqIdx[k]
            if (kprime-1) == (j+1):
                J12[j][k] = Pcomp(j+1)/V[j+1] + V[j+1]*np.real(yBus[j+1][j+1])
            else:
                J12[j][k] = V[j+1] *(np.real(yBus[j+1][kprime-1])*np.cos(theta[j+1]-theta[kprime-1]) + np.imag(yBus[j+1][kprime-1])*np.sin(theta[j+1]-theta[kprime-1]))
    return J12

def J21():
    J21 = np.zeros(shape=(numPQ, numBuses-1))
    
    for j in range(0,numPQ):
        kprime = pqIdx[j] 
        for k in range(0,numBuses-1): 
            if kprime-1 == k+1: 
                J21[j][k] = Pcomp(kprime-1) - (np.real(yBus[kprime-1][kprime-1])*V[kprime-1]**2)
            else: 
                J21[j][k] = ((-1) * V[kprime-1] * V[k+1]) * ((np.real(yBus[kprime-1][k+1]) * np.cos(theta[kprime-1] - theta[k+1])) + 
                                                          (np.imag(yBus[kprime-1][k+1]) * np.sin(theta[kprime-1] - theta[k+1])))

    return J21

def J22():
    J22 = np.zeros(shape=(numPQ, numPQ))
    
    for j in range(0,numPQ): 
        kprime1 = pqIdx[j]
        for k in range(0,numPQ): 
            kprime2 = pqIdx[k]
            if kprime1-1 == kprime2-1: 
                J22[j][k] = (Qcomp(kprime1-1)/V[kprime1-1])-(np.imag(yBus[kprime1-1][kprime1-1])*V[kprime1-1])
            else:
                J22[j][k] = V[kprime1-1]*((np.real(yBus[kprime1-1][kprime2-1])*np.sin(theta[kprime1-1]-theta[kprime2-1])) -
                                       (np.imag(yBus[kprime1-1][kprime2-1])*np.cos(theta[kprime1-1]-theta[kprime2-1]))) 
    return J22

def J():
    J1 = np.vstack((J11(),J21()))
    J2 = np.vstack((J12(),J22()))
    J = np.hstack((J1,J2))

    return J

def checkConverge(deltaP, deltaQ):
    for p in range(0, len(deltaP)): 
        if deltaP[p] > Pconv: 
            print(p)
            return False;
    for q in range(0, len(deltaQ)): 
        if deltaQ[q] > Qconv: 
            return False;
    return True;

def calcCorrection(jacobian, mismatch):
    invJacobian = np.linalg.inv(jacobian)
    correction = np.matmul(-invJacobian, mismatch)
    return correction;


def update(correction, numPQ):
    global V
    global theta 
    
    for thetaInd in range(0, len(correction) - numPQ): 
        theta[thetaInd+1] += correction[thetaInd][0]
    
    for vInd in range(0, numPQ):
        kprime = pqIdx[vInd]-1 
        V[kprime] += correction[len(theta)-1+vInd][0]
    return;

if __name__ == '__main__': 
    jacobian = J()
    
    P = deltaP()
    Q = deltaQ()
    
    dP = P.reshape(len(P),1)
    dQ = Q.reshape(len(Q),1)
    
    mismatch = np.vstack((dP,dQ))
    
    correction = calcCorrection(jacobian, mismatch)
    
    #print("correction:", correction)
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("dP", deltaP())
    print("dQ", deltaQ())
    #print("V:", V)
    #print("theta:", theta)
    #print("V length:", len(V))
    #print("theta length:", len(theta))
    #print("correction length:", len(correction))
    update(correction, numPQ)
    #print("numPQ", numPQ)
    #print("correction - numPQ", len(correction)-numPQ)
    #print("newV:", V)
    #print("newtheta", theta)
    print("newdP", deltaP())
    print("newdQ", deltaQ())
    print(checkConverge(deltaP(), deltaQ()))
    
    pass 


np.savetxt("Jacobian.csv", J(), delimiter=",")

b = np.asarray(deltaP())
np.savetxt("deltaP.csv", b, delimiter=",")

c = np.asarray(deltaQ())
np.savetxt("deltaQ.csv", c, delimiter=",")

d = np.asarray(J11())
np.savetxt("J11.csv", d, delimiter=",")

e = np.asarray(J12())
np.savetxt("J12.csv", e, delimiter=",")

f = np.asarray(J21())
np.savetxt("J21.csv", f, delimiter=",")

g = np.asarray(J22())
np.savetxt("J22.csv", g, delimiter=",")



#Calculate mismatch = [deltaP deltaQ]
#If mismatch is within limits, done abs(mismatch) < 0.1
#Build Jacobian
#corrections = -J[x0]^-1*Mismatch[x0]
#update: [delta[x1], v[x1]] = [delta[x0] v[x0]] + corrections

#repeat above steps until mismatches converge