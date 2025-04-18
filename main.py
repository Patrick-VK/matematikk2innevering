# jeg antar oppgaven vil at jeg skal programmere løsninger,
# derfor har jeg valgt å gjør dette

import numpy as py
from matplotlib import pyplot as plt
from matplotlib import animation as anim

# Oppgave 1)
def funcExp(x):
    return py.exp(x)
def dfdx(f,x,h):
    apprxDeriv = (f(x+h)-f(x))/h
    return apprxDeriv
def dfdx2(f,x,h):
    apprxDeriv = (f(x+h)-f(x-h))/(2*h)
    return apprxDeriv
def dfdx3(f,x,h):
    apprxDeriv = (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
    return apprxDeriv
def dudx(u,x,h,t):
    apprxDeriv2 = (u(x+h,t)-2*u(x,t)+u(x-h,t))/(h**2)
    return apprxDeriv2
def dudx2(u,x,h,t):
    apprxDeriv2 = (u(t)-2*u(t)+u(t))/(h**2)
    return apprxDeriv2
print("Oppgave 1) -------------------------------------------")
print("True value of f(x)=exp(x) when x=1.5 is: " + str(py.exp(1.5)) )
for i in range(2, 21): #fra 1 til 20
    h=1/(10**i)
    print(f"h={h} and f(x)=exp(x) when x=1.5 gives: "+str(dfdx(funcExp,1.5,h)) + "og har ett avvik på ca " + str(round(py.exp(1.5)-dfdx(funcExp,1.5,h), 5)))

#Når h<1/(10^9), da begynner feil resultater å oppbygge seg
#Når h<1/(10^15), da får man ikke realistiske resultater i det heletatt og "går åt skogen"
#At feilen er proposjonal med h kan observeres ved å se på avviket som reduseres
#Årsaken er at datamaskiner får enten lavere eller ingen presisjon ved veldig lave/høye tall
print("Oppgave 2) -------------------------------------------")
for i in range(2, 21): #fra 1 til 20
    h=1/(10**i)
    print(f"h={h} and f(x)=exp(x) when x=1.5 gives: "+str(dfdx2(funcExp,1.5,h)) + "og har ett avvik på ca " + str(round(py.exp(1.5)-dfdx2(funcExp,1.5,h), 5)))
#Når h<1/(10^14), da begynner feil resultater å oppbygge seg
#Når h<1/(10^(15)) da går h "åt skogen"
#O(h) i taylorrekken får lavere verdi fordi alle partall n av f^n(x) derivert blir fjernet
#I tillegg til at O(h) blir delt på 2h istedenfor h
#da blir O(h)=(f^3(x)/3!)h+(f^5(x)/5!)h^3+... 
#Med andre ord har alle kvadrater hos forrige O(h) blitt fjernet
#Dette impliserer at ligningen går raskere mot den riktige verdien
print("Oppgave 3) -------------------------------------------")
for i in range(2, 21): #fra 1 til 20
    h=1/(10**i)
    print(f"h={h} and f(x)=exp(x) when x=1.5 gives: "+str(dfdx3(funcExp,1.5,h)) + "og har ett avvik på ca " + str(round(py.exp(1.5)-dfdx3(funcExp,1.5,h), 5)))
#Her går avviket til ca 0 nesten med en gang
print("Oppgave 4) -------------------------------------------")
#For hver t_j regner vi ut alle x_i 
rows=30
cols=100 #n
#Ulike verdier av h og k gir ulik oppførsel (temperatur blir tapt om (k/(h**2)) < 0, eller motsatt)
h=2
k=1
#2D array u_ij
u_ij=[[0 for n in range(0,cols)] for n in range(0,rows)]
for i in range(1, cols-1):#initialkrav
    x_i=h*i
    u_ij[0][i]=py.sin(x_i)


for j in range(0, rows-1):
    t_j=k*j
    for i in range(1, cols-1):
        u_ij[j+1][i] = u_ij[j][i]+(k/(h**2))*(u_ij[j][i+1]-2*u_ij[j][i]+u_ij[j-1][i])


x=py.linspace(0,1,cols)
#Animasjon setup
fig, ax = plt.subplots()
plt.xlabel('x lengde')
plt.ylabel('temp')
yValues, = ax.plot(x, u_ij[0])


def update(i):
    yValues.set_ydata(u_ij[i])
    return yValues

ani = anim.FuncAnimation(fig, update, frames=len(u_ij), interval=200)
plt.show()
print("Oppgave 5) -------------------------------------------")

#Implisitt

u_ij=[[0 for n in range(0,cols)] for n in range(0,rows)]
for i in range(1, cols-1):#initialkrav
    x_i=h*i
    u_ij[0][i]=py.sin(x_i)

#Matrise
lambd=k/(h**2)
hoved_diag = (1 + 2*lambd) * py.ones(cols-2) #Kolonne med 1'ere, cols-2 pga randkravene, (1 + 2*lambd) pga (I+lambda*A). Dette blir en liste med hoved diagonalene 
side_diag  = -lambd * py.ones(cols-3) #Liste med side diagonalene, cols-3 pga side diagonal har ett mindre element
M = py.diag(hoved_diag) + py.diag(side_diag, 1) + py.diag(side_diag, -1) # lager en (cols-2)*(cols-2) matrise med hoved diagonal og side diagonaler

for j in range(0, rows-1):
    ind = [u_ij[j][i] for i in range(1, cols-1)]
    los = py.linalg.solve(M, ind)

    # Legg løsningene tilbake i u_ij
    for idx, val in enumerate(los, start=1):
        u_ij[j+1][idx] = val

x=py.linspace(0,1,cols)
#Animasjon setup
fig, ax = plt.subplots()
plt.xlabel('x lengde')
plt.ylabel('temp')
yValues, = ax.plot(x, u_ij[0])

def update(i):
    yValues.set_ydata(u_ij[i])
    return yValues

ani = anim.FuncAnimation(fig, update, frames=len(u_ij), interval=200)
plt.show()
print("Oppgave 6) -------------------------------------------")

#Crank–Nicolson

u_ij=[[0 for n in range(0,cols)] for n in range(0,rows)]
for i in range(1, cols-1):#initialkrav
    x_i=h*i
    u_ij[0][i]=py.sin(x_i)

hoved_diag = -2.0 * py.ones(cols-2)
side_diag  =  1.0 * py.ones(cols-3)
B = py.diag(hoved_diag) + py.diag(side_diag, 1) + py.diag(side_diag, -1) #trippel diagonal

M_v  = py.eye(cols-2) - (lambd/2.0) * B
M_h = py.eye(cols-2) + (lambd/2.0) * B

for j in range(rows-1):
    u_forr = py.array([u_ij[j][i] for i in range(1, cols-1)])
    b = M_h.dot(u_forr)
    los = py.linalg.solve(M_v, b)
    for indx, val in enumerate(los, start=1):
        u_ij[j+1][indx] = val

x=py.linspace(0,1,cols)

#Animasjon setup
fig, ax = plt.subplots()
plt.xlabel('x lengde')
plt.ylabel('temp')
yValues, = ax.plot(x, u_ij[0])

def update(i):
    yValues.set_ydata(u_ij[i])
    return yValues

ani = anim.FuncAnimation(fig, update, frames=len(u_ij), interval=200)
plt.show()