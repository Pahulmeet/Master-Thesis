from application import app
from flask import render_template, url_for, request
import pandas as pd
import json
import plotly
import plotly.express as px

import torch
import torch.nn as nn
import numpy as np
import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

res = []
@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    print(output)
    result = json.loads(output)
    global res
    res.clear()
    print(result)
    if(result['state1'] == "NO S ARE M" or result['state1'] == 'NO M ARE S'):
        res.append(1)
    elif(result['state1'] == 'SOME S ARE M' or result['state1'] == 'SOME M ARE S'):
        res.append(2)
    elif(result['state1'] == 'ALL S ARE M'):
        res.append(3)
    elif(result['state1'] == 'ALL M ARE S'):
        res.append(5)
    elif(result['state1'] == 'SOME S ARE NOT M'):
        res.append(4)
    elif(result['state1'] == 'SOME M ARE NOT S'):
        res.append(41)
    
    if(result['state2'] == "NO M ARE P" or result['state2'] == 'NO P ARE M'):
        res.append(1)
    elif(result['state2'] == 'SOME M ARE P' or result['state2'] == 'SOME P ARE M'):
        res.append(2)
    elif(result['state2'] == 'ALL M ARE P'):
        res.append(3)
    elif(result['state2'] == 'ALL P ARE M'):
        res.append(5)
    elif(result['state2'] == 'SOME M ARE NOT P'):
        res.append(4)
    elif(result['state2'] == 'SOME P ARE NOT M'):
        res.append(41)

    if(result['state3'] == "NO S ARE P" or result['state3'] == 'NO P ARE S'):
        res.append(1)
    elif(result['state3'] == 'SOME S ARE P' or result['state3'] == 'SOME P ARE S'):
        res.append(2)
    elif(result['state3'] == 'ALL S ARE P'):
        res.append(3)
    elif(result['state3'] == 'ALL P ARE S'):
        res.append(5)
    elif(result['state3'] == 'SOME S ARE NOT P'):
        res.append(4)
    elif(result['state3'] == 'SOME P ARE NOT S'):
        res.append(41)
    return result

rho = 1 # Radius of the Sphere
theta = 0 # Angle on the X-Y Plane between [0,2*pi]
phi = 0 # ANgle on the Z-Axis between [0, pi]

main_count = 0 
q1 = np.array([])
q2 = np.array([])
q3 = np.array([])
angles1 = torch.tensor([])
angles2 = torch.tensor([])
angles3 = torch.tensor([])
rel12 = 0
rel23 = 0
rel13 = 0
maxIter = 3000
maxRounds = 1

lr1 = 0.001
lr2 = 0.01

l12 = 1
l23 = 1
l13 = 1

def distance_between_centres(c1,c2):
    d = rho*np.arccos((c1[0].item()*c2[0].item() + c1[1].item()*c2[1].item() + c1[2].item()*c2[2].item())/(rho*rho))
    return d

def distance_between_points(c1,dot_x,dot_y,dot_z):
    d = rho*np.arccos((c1[0].item()*dot_x + c1[1].item()*dot_y + c1[2].item()*dot_z)/(rho*rho))
    return d

@app.route("/")
def index():
    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    #random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([45])
    global q1, r1
    q1 = np.append(random_theta, random_phi)
    q1 = np.append(q1, random_alpha)
    x1 = rho*np.sin(random_phi)*np.cos(random_theta)
    y1 = rho*np.sin(random_phi)*np.sin(random_theta)
    z1 = rho*np.cos(random_phi)
    r1 = rho*np.pi*(random_alpha/360) # Radius of Chrome
    l1 = [x1,y1,z1,r1]
    c1 = torch.tensor(l1, dtype = torch.float32, requires_grad = True)

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    #random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([45])
    global q2, r2
    q2 = np.append(random_theta, random_phi)
    q2 = np.append(q2, random_alpha)
    x2 = rho*np.sin(random_phi)*np.cos(random_theta)
    y2 = rho*np.sin(random_phi)*np.sin(random_theta)
    z2 = rho*np.cos(random_phi)
    r2 = rho*np.pi*(random_alpha/360) # Radius of Chrome
    l2 = [x2,y2,z2,r2]
    c2 = torch.tensor(l2, dtype = torch.float32, requires_grad = True)

    temp_df1 = pd.DataFrame(l1)
    temp_df1 = temp_df1.T
    temp_df2 = pd.DataFrame(l2)
    temp_df2 = temp_df2.T
    df = temp_df1.append(temp_df2)
    df.columns =['X_axis', 'Y_axis', 'Z_axis', 'Radius']

    global json_data 
    json_data = df.to_dict(orient='records')

    return render_template('index.html', title = 'PythonIsHere!', JSON_data = json_data)

@app.route("/animate")
def animate():
    rounds = 0
    gLoss = 1
    maxIter = 1000
    lr = 0.001
    TB = np.array([1])
    loss = nn.MSELoss()
    angles1 = torch.from_numpy(q1)
    angles1.requires_grad = True
    angles2 = torch.from_numpy(q2)
    angles2.requires_grad = True
    sum_of_radii = r1+r2-0.01
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    def RESU_D2O_new(angles1,angles2):
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))   
        r1 = rho*np.pi*(q1[2]/360)
        r2 = rho*np.pi*(q2[2]/360)
        #print(angles1)
        return max(0, d - (r1+r2))

    print(angles1)
    print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    while(rounds < maxIter and gLoss > 0 and RESU_D2O_new(angles1,angles2)):
        #print(angles1)
        for k in range(1):
            #print("in k loop")
            for j in range(1):
                #print("in j loop")
                z=0
                print(angles1)
                #while(RESU_D2O_new(angles1,angles2) == 0 or z <= maxIter):
                while(z <= maxIter and RESU_D2O_new(angles1,angles2)):
                    #print(angles1, "---", angles2)
                    with torch.no_grad():
                        tp1 = angles1
                        tp2 = angles2
                        np_arr1 = tp1.numpy()
                        np_arr2 = tp2.numpy()
                        ls1 = []
                        ls2 = []
                        ls1 = np_arr1.tolist()
                        ls2 = np_arr2.tolist()
                        save_1.append(ls1)
                        save_2.append(ls2)
                    print("hhhhhhhhhh == ", z, "------",np_arr1)
                    d = forward()
                    d = d.type(torch.FloatTensor)
                    tr = tr.type(torch.FloatTensor)
                    l = loss(d,tr)
                    l.backward()
                    with torch.no_grad():
                        angles1 -= lr*angles1.grad
                        angles2 -= lr*angles2.grad
                    angles1.grad.zero_()
                    angles2.grad.zero_()
                    z = z+1
                    print("Value of d : ", d)
                    print("Value of loss : ", l)
    #print(save_1)
    df1 = pd.DataFrame(save_1, columns = ['theta1','phi1','alpha1'])
    df2 = pd.DataFrame(save_2, columns = ['theta2','phi2','alpha2'])
    df_concat = pd.concat([df1, df2], axis=1)
    #print(df1.head())
    #new_df.columns =['X_axis', 'Y_axis', 'Z_axis', 'Radius']
    json_df_concat = df_concat.to_dict(orient='records')
    #json_df2 = df2.to_dict(orient='records')
    return render_template('animate.html', title = 'PythonIsHere!', JSON_data = json_df_concat)

@app.route("/part_of")
def animate_part_of():
    print('here!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    rounds = 0
    gLoss = 1
    maxIter = 1000
    lr = 0.001
    loss = nn.MSELoss()
    def loss_part_of(r1,d,r2):
        loss_p = r2 + d - r1
        return loss_p
    angles1 = torch.from_numpy(q1)
    angles1.requires_grad = True
    angles2 = torch.from_numpy(q2)
    angles2.requires_grad = True
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        r1 = rho*angles1[2]
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*angles2[2]
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d,r1,r2
    def RESU_O2P_new(angles1,angles2):
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        r1 = q1[2]
        r2 = q2[2]
        return max(0, d +(r2-r1))

    print(angles1)
    print(angles2)
    save_1 = []
    save_2 = []
    print(angles1)
    print(angles2)
    while(rounds < maxIter and gLoss > 0 and RESU_O2P_new(angles1,angles2)):
        #print(angles1)
        for k in range(1):
            #print("in k loop")
            for j in range(1):
                #print("in j loop")
                z=0
                print(angles1)
                
                #print("in TB if loop")

                #print('print old radii', r1, 'and', r2)
                #print('print new radii', rad1, 'and', rad2)

                """sum_of_radii1 = rad1+rad2
                tr1 = torch.tensor([sum_of_radii1], dtype=torch.float32) # For disconnected to overlap

                if(rad2>rad1):
                    sum_of_radii2 = rad2-rad1
                else:
                    sum_of_radii2 = rad1-rad2
                tr2 = torch.tensor([sum_of_radii2], dtype=torch.float32) # to part of"""


                print("Alpha Angles: ", angles1[2], angles2[2])
                
                while(RESU_O2P_new(angles1,angles2) and z <= maxIter):
                    with torch.no_grad():
                        tp1 = angles1
                        tp2 = angles2
                        np_arr1 = tp1.numpy()
                        np_arr2 = tp2.numpy()
                        ls1 = []
                        ls2 = []
                        ls1 = np_arr1.tolist()
                        ls2 = np_arr2.tolist()
                        save_1.append(ls1)
                        save_2.append(ls2)

                    d,r1,r2 = forward()
                    d = d.type(torch.FloatTensor)


                    #tr2 = tr2.type(torch.FloatTensor)

                    l = loss_part_of(r1,d,r2)
                    l.backward()

                    with torch.no_grad():
                        angles1 -= lr*angles1.grad
                        angles2 -= lr*angles2.grad
                    #print("angles1 grad", angles1.grad)
                    #print("angles2 grad", angles1.grad)
                    angles1.grad.zero_()
                    angles2.grad.zero_()
                    z = z+1
                    print("Value of d2 : ", d)
                    print(z)
                    print("RESU_O2P_new(angles1,angles2)", RESU_O2P_new(angles1,angles2))



    df1 = pd.DataFrame(save_1, columns = ['theta1','phi1','alpha1'])
    df2 = pd.DataFrame(save_2, columns = ['theta2','phi2','alpha2'])
    df_concat = pd.concat([df1, df2], axis=1)
    #print(df1.head())
    #new_df.columns =['X_axis', 'Y_axis', 'Z_axis', 'Radius']
    json_df_concat = df_concat.to_dict(orient='records')
    #json_df2 = df2.to_dict(orient='records')
    return render_template('part_of.html', title = 'PythonIsHere!', JSON_data = json_df_concat)


@app.route("/d2c")
def d2c():
    rounds = 0
    gLoss = 1
    maxIter = 1000
    lr = 0.001
    TB = np.array([1])
    loss = nn.MSELoss()
    angles1 = torch.from_numpy(q1)
    angles1.requires_grad = True
    angles2 = torch.from_numpy(q2)
    angles2.requires_grad = True
    sum_of_radii = r1+r2-0.01
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    def RESU_D2O_new(angles1,angles2):
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))   
        r1 = rho*np.pi*(q1[2]/360)
        r2 = rho*np.pi*(q2[2]/360)
        #print(angles1)
        return max(0, d - (r1+r2))

    print(angles1)
    print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    while(rounds < maxIter and gLoss > 0 and RESU_D2O_new(angles1,angles2)):
        #print(angles1)
        for k in range(1):
            #print("in k loop")
            for j in range(1):
                #print("in j loop")
                z=0
                print(angles1)
                #while(RESU_D2O_new(angles1,angles2) == 0 or z <= maxIter):
                while(z <= maxIter and RESU_D2O_new(angles1,angles2)):
                    #print(angles1, "---", angles2)
                    with torch.no_grad():
                        tp1 = angles1
                        tp2 = angles2
                        np_arr1 = tp1.numpy()
                        np_arr2 = tp2.numpy()
                        ls1 = []
                        ls2 = []
                        ls1 = np_arr1.tolist()
                        ls2 = np_arr2.tolist()
                        save_1.append(ls1)
                        save_2.append(ls2)
                    print("hhhhhhhhhh == ", z, "------",np_arr1)
                    d = forward()
                    d = d.type(torch.FloatTensor)
                    tr = tr.type(torch.FloatTensor)
                    l = loss(d,tr)
                    l.backward()
                    with torch.no_grad():
                        angles1 -= lr*angles1.grad
                        angles2 -= lr*angles2.grad
                    angles1.grad.zero_()
                    angles2.grad.zero_()
                    z = z+1
                    print("Value of d : ", d)
                    print("Value of loss : ", l)
    
    fx1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    fy1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    fz1 = rho*torch.cos(angles1[1])
    fx2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    fy2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    fz2 = rho*torch.cos(angles2[1])

    l1_new = [fx1.item(), fy1.item(), fz1.item(), r1[0]]
    l2_new = [fx2.item(), fy2.item(), fz2.item(), r2[0]]
    temp_df1 = pd.DataFrame(l1_new)
    temp_df1 = temp_df1.T
    temp_df2 = pd.DataFrame(l2_new)
    temp_df2 = temp_df2.T
    new_df = temp_df1.append(temp_df2)
    new_df.columns =['X_axis', 'Y_axis', 'Z_axis', 'Radius']
    json_data1 = new_df.to_dict(orient='records')
    return render_template('d2c.html', title = 'PythonIsHere!', JSON_data = json_data1)


def distance_between_centres(c1,c2):
    d = rho*np.arccos((c1[0].item()*c2[0].item() + c1[1].item()*c2[1].item() + c1[2].item()*c2[2].item())/(rho*rho))
    return d
def check_disconnect(d, c1, c2):
    chk = 1000
    if(d - (c1[3].item() + c2[3].item()) >= 0):
        chk = 1
    else:
        chk = 0
    return chk
def check_overlap(d, c1, c2):
    chk = 1000
    if(d > abs(c1[3].item() - c2[3].item()) and d < abs(c1[3].item() + c2[3].item())):
        chk = 1
    else:
        chk = 0
    return chk
def check_part_of(d, c1, c2):
    chk = 1000
    if(d + c1[3].item() <= c2[3].item()):
        chk = 1
    else:
        chk = 0
    return chk

tolerance1 = 0.14 #originally was 0.01\n",
tolerance2 = 0.1 #originally was 0.03\n",

tolerance3 = 0.15 #origunally was 0.02 (should be slightly more than tolerance1 in training function)\n",
divide_by = 5.5

def check_relation(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1]) 
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    chk = 1000
    if(d.item() - (r1 + r2) >= 0):
        chk = 1
        #print('1: Disconnect, no X are Y')
    elif(d.item() + r1 <= r2):
        chk = 3
        #print('3: First Chrome Part of Second Chrome, all X are Y')
    elif(d.item() + r2 <= r1):
        chk = 5
        #print('4: Second Chrome Part of First Chrome')
    elif(d.item() - (r1 + r2) < 0):
        chk = 2
        #print('2: Partial Overlap, some X are Y')
    elif(d.item() + r1 > r2):
        chk = 4
    elif(d.item() + r2 > r1):
        chk = 41
        #print('4: Second Chrome Part of First Chrome, some Y are not X')
    return chk
    

def distance_between_points(c1,dot_x,dot_y,dot_z):
    d = rho*np.arccos((c1[0].item()*dot_x + c1[1].item()*dot_y + c1[2].item()*dot_z)/(rho*rho))
    return d

def RESU_D2O_new(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1]) 
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    return max(0, d.item() - (r1+r2-tolerance1))

def RESU_O2D_new(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1]) 
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    return min(0, d.item() - (r1+r2+tolerance1))

def RESU_O2P_new(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1])
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    return max(0, r2+d+tolerance1-r1)

def RESU_P2O_new(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1])
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    return min(0, d.item()-r2-r1+tolerance2)

def RESU_O2IP_new(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1])
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    return max(0, r1+d.item()+tolerance1-r2)

def RESU_IP2O_new(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1])
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    return min(0, d.item()-r2-r1+tolerance2)

def print_all(angles1,angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1]) 
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    print("Co-ordinates of C1:", x1.item(),y1.item(),z1.item())
    print("Co-ordinates of C2:", x2.item(),y2.item(),z2.item())
    print("Radius of C1:", r1)
    print("Radius of C2:", r2)
    print("Sum of Radii:", r1+r2, " and Distance between them:", d.item())


rho = 1 # Radius of the Sphere
theta = 0 # Angle on the X-Y Plane between [0,2*pi]
phi = 0 # Angle on the Z-Axis between [0, pi]
chrome_angle = 70
def create_chromes():
    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    #random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([chrome_angle])
    q1 = np.append(random_theta, random_phi)
    q1 = np.append(q1, random_alpha)
    x1 = rho*np.sin(random_phi)*np.cos(random_theta)
    y1 = rho*np.sin(random_phi)*np.sin(random_theta)
    z1 = rho*np.cos(random_phi)
    r1 = rho*np.pi*(random_alpha/360) # Radius of Chrome
    l1 = [x1,y1,z1,r1]
    c1 = torch.tensor(l1, dtype = torch.float32, requires_grad = True)
    #print("Chrome 1 Co-ordinates and radius (x,y,z,r) : ", "x:", c1[0].item()," y:",c1[1].item()," z:",c1[2].item()," r:",c1[3].item())

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    #random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([chrome_angle])
    q2 = np.append(random_theta, random_phi)
    q2 = np.append(q2, random_alpha)
    x2 = rho*np.sin(random_phi)*np.cos(random_theta)
    y2 = rho*np.sin(random_phi)*np.sin(random_theta)
    z2 = rho*np.cos(random_phi)
    r2 = rho*np.pi*(random_alpha/360) # Radius of Chrome
    l2 = [x2,y2,z2,r2]
    c2 = torch.tensor(l2, dtype = torch.float32, requires_grad = True)
    #print("Chrome 2 Co-ordinates and radius (x,y,z,r) : ", "x:", c2[0].item()," y:",c2[1].item()," z:",c2[2].item()," r:",c2[3].item())

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    #random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([chrome_angle])
    q3 = np.append(random_theta, random_phi)
    q3 = np.append(q3, random_alpha)
    x3 = rho*np.sin(random_phi)*np.cos(random_theta)
    y3 = rho*np.sin(random_phi)*np.sin(random_theta)
    z3 = rho*np.cos(random_phi)
    r3 = rho*np.pi*(random_alpha/360) # Radius of Chrome
    l3 = [x3,y3,z3,r3]
    c3 = torch.tensor(l3, dtype = torch.float32, requires_grad = True)
    #print("Chrome 3 Co-ordinates and radius (x,y,z,r) : ", "x:", c3[0].item()," y:",c3[1].item()," z:",c3[2].item()," r:",c3[3].item())
    return q1, q2, q3



def run12(relation):
    global l12
    global angles1,angles2,angles3,rel12,rel23,rel13
    if(relation == 1):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 2):
            Overlap2Disconnect(angles1, angles2)
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 3):
            InversePartof2Overlap(angles1, angles2)
            Overlap2Disconnect(angles1, angles2)
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 4):
            Overlap2Disconnect(angles1, angles2)
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 41):
            Overlap2Disconnect(angles1, angles2)
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 5):
            Partof2Overlap(angles1, angles2)
            Overlap2Disconnect(angles1, angles2)
            l12 = abs(RESU_O2D_new(angles1,angles2))
        else:
            #print('Disconnect relation satisfied')
            l12 = 0
    if(relation == 2):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 1):
            Disconnect2Overlap(angles1, angles2)
            l12 = abs(RESU_D2O_new(angles1,angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l12 = 0
    if(relation == 3):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 1):
            Disconnect2Overlap(angles1, angles2)
            Overlap2InversePartof(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 2):
            Overlap2InversePartof(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 4):
            Overlap2InversePartof(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 41):
            Overlap2InversePartof(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 5):
            Partof2Overlap(angles1, angles2)
            Overlap2InversePartof(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        else:
            #print('Inverse Part of relation satisfied')
            l12 = 0
    if(relation == 4):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 3):
            InversePartof2Overlap(angles1, angles2)
            l12 = abs(RESU_IP2O_new(angles1,angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l12 = 0
    if(relation == 41):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 5):
            Partof2Overlap(angles1, angles2)
            l12 = abs(RESU_P2O_new(angles1,angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l12 = 0
    if(relation == 5):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 1):
            Disconnect2Overlap(angles1, angles2)
            Overlap2Partof(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 2):
            Overlap2Partof(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 3):
            InversePartof2Overlap(angles1, angles2)
            Overlap2Partof(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 4):
            Overlap2Partof(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 41):
            Overlap2Partof(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l12 = abs(RESU_O2P_new(angles1,angles2))
        else:
            #print('Part of relation satisfied')
            l12 = 0
            
def run23(relation):
    global l23
    global angles1,angles2,angles3,rel12,rel23,rel13
    if(relation == 1):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 2):
            Overlap2Disconnect2(angles2, angles3)
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 3):
            InversePartof2Overlap2(angles2, angles3)
            Overlap2Disconnect2(angles2, angles3)
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 4):
            Overlap2Disconnect2(angles2, angles3)
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 41):
            Overlap2Disconnect2(angles2, angles3)
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 5):
            Partof2Overlap2(angles2, angles3)
            Overlap2Disconnect2(angles2, angles3)
            l23 = abs(RESU_O2D_new(angles2,angles3))
        else:
            #print('Disconnect relation satisfied')
            l23 = 0
    if(relation == 2):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            Disconnect2Overlap2(angles2, angles3)
            l23 = abs(RESU_D2O_new(angles2,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 3):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            Disconnect2Overlap2(angles2, angles3)
            Overlap2InversePartof2(angles2, angles3)
            with torch.no_grad():
                angles2[2] = angles3[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 2):
            Overlap2InversePartof2(angles2, angles3)
            with torch.no_grad():
                angles2[2] = angles3[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 4):
            Overlap2InversePartof2(angles2, angles3)
            with torch.no_grad():
                angles2[2] = angles3[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 41):
            Overlap2InversePartof2(angles2, angles3)
            with torch.no_grad():
                angles2[2] = angles3[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 5):
            Partof2Overlap2(angles2, angles3)
            Overlap2InversePartof2(angles2, angles3)
            with torch.no_grad():
                angles2[2] = angles3[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        else:
            #print('Inverse Part of relation satisfied')
            l23 = 0
    if(relation == 4):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 3):
            InversePartof2Overlap2(angles2, angles3)
            l23 = abs(RESU_IP2O_new(angles2,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 41):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 5):
            Partof2Overlap2(angles2, angles3)
            l23 = abs(RESU_P2O_new(angles2,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 5):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            Disconnect2Overlap2(angles2, angles3)
            Overlap2Partof2(angles2, angles3)
            with torch.no_grad():
                angles3[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 2):
            Overlap2Partof2(angles2, angles3)
            with torch.no_grad():
                angles3[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 3):
            InversePartof2Overlap2(angles2, angles3)
            Overlap2Partof2(angles2, angles3)
            with torch.no_grad():
                angles3[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 4):
            Overlap2Partof2(angles2, angles3)
            with torch.no_grad():
                angles3[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 4):
            Overlap2Partof2(angles2, angles3)
            with torch.no_grad():
                angles3[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2P_new(angles2,angles3))
        else:
            #print('Part of relation satisfied')
            l23 = 0
            
def run13(relation):
    global l13
    global angles1,angles2,angles3,rel12,rel23,rel13
    temp_relation = rel23
    run23(temp_relation) # Problem, correct relation
    if(relation == 1):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 2):
            Overlap2Disconnect(angles1, angles3)
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 3):
            InversePartof2Overlap(angles1, angles3)
            Overlap2Disconnect(angles1, angles3)
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 4):
            Overlap2Disconnect(angles1, angles3)
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 41):
            Overlap2Disconnect(angles1, angles3)
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 5):
            Partof2Overlap(angles1, angles3)
            Overlap2Disconnect(angles1, angles3)
            l13 = abs(RESU_O2D_new(angles1,angles3))
        else:
            #print('Disconnect relation satisfied')
            l13 = 0
    if(relation == 2):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 1):
            Disconnect2Overlap(angles1, angles3)
            l13 = abs(RESU_D2O_new(angles1,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l13 = 0
    if(relation == 3):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 1):
            Disconnect2Overlap(angles1, angles3)
            Overlap2InversePartof(angles1, angles3)
            with torch.no_grad():
                angles1[2] = angles3[2]/divide_by
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 2):
            Overlap2InversePartof(angles1, angles3)
            with torch.no_grad():
                angles1[2] = angles3[2]/divide_by
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 4):
            Overlap2InversePartof(angles1, angles3)
            with torch.no_grad():
                angles1[2] = angles3[2]/divide_by
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 41):
            Overlap2InversePartof(angles1, angles3)
            with torch.no_grad():
                angles1[2] = angles3[2]/divide_by
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 5):
            Partof2Overlap(angles1, angles3)
            Overlap2InversePartof(angles1, angles3)
            with torch.no_grad():
                angles1[2] = angles3[2]/divide_by
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        else:
            #print('Inverse Part of relation satisfied')
            l13 = 0
    if(relation == 4):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 3):
            InversePartof2Overlap(angles1, angles3)
            l13 = abs(RESU_IP2O_new(angles1,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l13 = 0
    if(relation == 41):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 5):
            Partof2Overlap(angles1, angles3)
            l13 = abs(RESU_P2O_new(angles1,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l13 = 0
    if(relation == 5):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 1):
            Disconnect2Overlap(angles1, angles3)
            Overlap2Partof(angles1, angles3)
            with torch.no_grad():
                angles3[2] = angles1[2]/divide_by
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 2):
            Overlap2Partof(angles1, angles3)
            with torch.no_grad():
                angles3[2] = angles1[2]/divide_by
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 3):
            InversePartof2Overlap(angles1, angles3)
            Overlap2Partof(angles1, angles3)
            with torch.no_grad():
                angles3[2] = angles1[2]/divide_by
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 4):
            Overlap2Partof(angles1, angles3)
            with torch.no_grad():
                angles3[2] = angles1[2]/divide_by
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 41):
            Overlap2Partof(angles1, angles3)
            with torch.no_grad():
                angles3[2] = angles1[2]/divide_by
            l13 = abs(RESU_O2P_new(angles1,angles3))
        else:
            #print('Part of relation satisfied')
            l13 = 0



def Overlap2Disconnect(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2+tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d
    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []

    z=0
    while(z <= maxIter and RESU_O2D_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            #angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        #angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2D_new(angles1,angles2))  """ 

def Overlap2Disconnect2(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2+tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d
    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    z=0
    while(z <= maxIter and RESU_O2D_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            #angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        #angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2D_new(angles1,angles2))   """
          

def InversePartof2Overlap(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2-tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_IP2O_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            #angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        #angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_IP2O_new(angles1,angles2))"""


def InversePartof2Overlap2(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2-tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_IP2O_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            #angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        #angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_IP2O_new(angles1,angles2))"""


def Overlap2InversePartof(angles1,angles2):
    lr = lr2
    def loss_part_of(r1,d,r2):
        loss_p = r1 + d + tolerance3 - r2
        return loss_p
    def forward1():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d,r1,r2

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_O2IP_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d,r1,r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1,d,r2)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            #angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        #angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2IP_new(angles1,angles2))"""
        

def Overlap2InversePartof2(angles1,angles2):
    lr = lr2
    def loss_part_of(r1,d,r2):
        loss_p = r1 + d + tolerance3 - r2
        return loss_p
    def forward1():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d,r1,r2

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_O2IP_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d,r1,r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1,d,r2)
        l.backward()
        with torch.no_grad():
            #angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        #angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2IP_new(angles1,angles2))"""
        


def Partof2Overlap(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2-tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_P2O_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            #angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        #angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_P2O_new(angles1,angles2))"""


def Partof2Overlap2(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2-tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_P2O_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            #angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        #angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_P2O_new(angles1,angles2))"""


def Overlap2Partof(angles1,angles2):
    lr = lr2
    def loss_part_of(r1,d,r2):
        loss_p = r2 + d + tolerance3 - r1
        return loss_p
    def forward1():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d,r1,r2

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    z=0
    while(z <= maxIter and RESU_O2P_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d,r1,r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1,d,r2)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            #angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        #angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2P_new(angles1,angles2))"""
        


def Overlap2Partof2(angles1,angles2):
    lr = lr2
    def loss_part_of(r1,d,r2):
        loss_p = r2 + d + tolerance3 - r1
        return loss_p
    def forward1():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1])
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d,r1,r2

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    z=0
    while(z <= maxIter and RESU_O2P_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d,r1,r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1,d,r2)
        l.backward()
        with torch.no_grad():
            #angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        #angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2P_new(angles1,angles2))"""
        
def Disconnect2Overlap(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2-tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_D2O_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            #angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        #angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_D2O_new(angles1,angles2))"""


def Disconnect2Overlap2(angles1,angles2):
    lr = lr1
    loss = nn.MSELoss()
    rad1 = rho*np.pi*(angles1[2].item()/360)
    rad2 = rho*np.pi*(angles2[2].item()/360)
    sum_of_radii = rad1+rad2-tolerance3
    tr = torch.tensor([sum_of_radii], dtype=torch.float32)
    def forward():
        x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
        y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
        z1 = rho*torch.cos(angles1[1]) 
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d

    #print(angles1)
    #print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    
    z=0
    while(z <= maxIter and RESU_D2O_new(angles1,angles2)):
        with torch.no_grad():
            tp1 = angles1
            tp2 = angles2
            np_arr1 = tp1.numpy()
            np_arr2 = tp2.numpy()
            ls1 = []
            ls2 = []
            ls1 = np_arr1.tolist()
            ls2 = np_arr2.tolist()
            save_1.append(ls1)
            save_2.append(ls2)
        d = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d,tr)
        l.backward()
        with torch.no_grad():
            #angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        #angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_D2O_new(angles1,angles2))"""




def loss12(relation):
    global l12
    global angles1,angles2,angles3,rel12,rel23,rel13
    if(relation == 1):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 2):
            
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 3):
            
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 4):
            
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 41):
            
            l12 = abs(RESU_O2D_new(angles1,angles2))
        elif(chk == 5):
            
            l12 = abs(RESU_O2D_new(angles1,angles2))
        else:
            #print('Disconnect relation satisfied')
            l12 = 0
    if(relation == 2):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 1):
            
            l12 = abs(RESU_D2O_new(angles1,angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l12 = 0
    if(relation == 3):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 1):
            
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 2):
            
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 4):
            
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 41):
            
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        elif(chk == 5):
            
            l12 = abs(RESU_O2IP_new(angles1,angles2))
        else:
            #print('Inverse Part of relation satisfied')
            l12 = 0
    if(relation == 4):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 3):
            
            l12 = abs(RESU_IP2O_new(angles1,angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l12 = 0
    if(relation == 41):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 5):
            
            l12 = abs(RESU_P2O_new(angles1,angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l12 = 0
    if(relation == 5):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l12 = 1
        if(chk == 1):
            
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 2):
            
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 3):
            
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 4):
            
            l12 = abs(RESU_O2P_new(angles1,angles2))
        elif(chk == 41):
            
            l12 = abs(RESU_O2P_new(angles1,angles2))
        else:
            #print('Part of relation satisfied')
            l12 = 0
            
def loss23(relation):
    global l23
    global angles1,angles2,angles3,rel12,rel23,rel13
    if(relation == 1):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 2):
            
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 3):
            
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 4):
            
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 41):
            
            l23 = abs(RESU_O2D_new(angles2,angles3))
        elif(chk == 5):
            
            l23 = abs(RESU_O2D_new(angles2,angles3))
        else:
            #print('Disconnect relation satisfied')
            l23 = 0
    if(relation == 2):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            
            l23 = abs(RESU_D2O_new(angles2,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 3):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 2):
            
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 4):
            
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 41):
            
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        elif(chk == 5):
            
            l23 = abs(RESU_O2IP_new(angles2,angles3))
        else:
            #print('Inverse Part of relation satisfied')
            l23 = 0
    if(relation == 4):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 3):
            
            l23 = abs(RESU_IP2O_new(angles2,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 41):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 5):
            
            l23 = abs(RESU_P2O_new(angles2,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 5):
        chk = check_relation(angles2, angles3)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 2):
            
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 3):
            
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 4):
            
            l23 = abs(RESU_O2P_new(angles2,angles3))
        elif(chk == 41):
            
            l23 = abs(RESU_O2P_new(angles2,angles3))
        else:
            #print('Part of relation satisfied')
            l23 = 0
            
def loss13(relation):
    global l13
    global angles1,angles2,angles3,rel12,rel23,rel13
    temp_relation = rel23
    run23(temp_relation) # Problem, correct relation
    if(relation == 1):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 2):
            
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 3):
            
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 4):
            
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 41):
            
            l13 = abs(RESU_O2D_new(angles1,angles3))
        elif(chk == 5):
            
            l13 = abs(RESU_O2D_new(angles1,angles3))
        else:
            #print('Disconnect relation satisfied')
            l13 = 0
    if(relation == 2):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 1):
            
            l13 = abs(RESU_D2O_new(angles1,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l13 = 0
    if(relation == 3):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 1):
            
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 2):
            
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 4):
            
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 41):
            
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        elif(chk == 5):
            
            l13 = abs(RESU_O2IP_new(angles1,angles3))
        else:
            #print('Inverse Part of relation satisfied')
            l13 = 0
    if(relation == 4):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 3):
            
            l13 = abs(RESU_IP2O_new(angles1,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l13 = 0
    if(relation == 41):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 5):
            
            l13 = abs(RESU_P2O_new(angles1,angles3))
        else:
            #print('Partial Overlap relation satisfied')
            l13 = 0
    if(relation == 5):
        chk = check_relation(angles1, angles3)
        if(chk != relation):
            l13 = 1
        if(chk == 1):
            
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 2):
            
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 3):
            
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 4):
            
            l13 = abs(RESU_O2P_new(angles1,angles3))
        elif(chk == 41):
            
            l13 = abs(RESU_O2P_new(angles1,angles3))
        else:
            #print('Part of relation satisfied')
            l13 = 0


            
            
def run12_1fix(relation):
    global l12
    global angles1,angles2,angles3,rel12,rel23,rel13
    if(relation == 1):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l23 = 1
        if(chk == 2):
            Overlap2Disconnect2(angles1, angles2)
            l23 = abs(RESU_O2D_new(angles1, angles2))
        elif(chk == 3):
            InversePartof2Overlap2(angles1, angles2)
            Overlap2Disconnect2(angles1, angles2)
            l23 = abs(RESU_O2D_new(angles1, angles2))
        if(chk == 4):
            Overlap2Disconnect2(angles1, angles2)
            l23 = abs(RESU_O2D_new(angles1, angles2))
        if(chk == 41):
            Overlap2Disconnect2(angles1, angles2)
            l23 = abs(RESU_O2D_new(angles1, angles2))
        elif(chk == 5):
            Partof2Overlap2(angles1, angles2)
            Overlap2Disconnect2(angles1, angles2)
            l23 = abs(RESU_O2D_new(angles1, angles2))
        else:
            #print('Disconnect relation satisfied')
            l23 = 0
    if(relation == 2):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            Disconnect2Overlap2(angles1, angles2)
            l23 = abs(RESU_D2O_new(angles1, angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 3):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            Disconnect2Overlap2(angles1, angles2)
            Overlap2InversePartof2(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles1, angles2))
        elif(chk == 2):
            Overlap2InversePartof2(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles1, angles2))
        elif(chk == 4):
            Overlap2InversePartof2(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles1, angles2))
        elif(chk == 41):
            Overlap2InversePartof2(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles1, angles2))
        elif(chk == 5):
            Partof2Overlap2(angles1, angles2)
            Overlap2InversePartof2(angles1, angles2)
            with torch.no_grad():
                angles1[2] = angles2[2]/divide_by
            l23 = abs(RESU_O2IP_new(angles1, angles2))
        else:
            #print('Inverse Part of relation satisfied')
            l23 = 0
    if(relation == 4):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l23 = 1
        if(chk == 3):
            InversePartof2Overlap2(angles1, angles2)
            l23 = abs(RESU_IP2O_new(angles1, angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 41):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l23 = 1
        if(chk == 5):
            Partof2Overlap2(angles1, angles2)
            l23 = abs(RESU_P2O_new(angles1, angles2))
        else:
            #print('Partial Overlap relation satisfied')
            l23 = 0
    if(relation == 5):
        chk = check_relation(angles1, angles2)
        if(chk != relation):
            l23 = 1
        if(chk == 1):
            Disconnect2Overlap2(angles1, angles2)
            Overlap2Partof2(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l23 = abs(RESU_O2P_new(angles1, angles2))
        elif(chk == 2):
            Overlap2Partof2(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l23 = abs(RESU_O2P_new(angles1, angles2))
        elif(chk == 3):
            InversePartof2Overlap2(angles1, angles2)
            Overlap2Partof2(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l23 = abs(RESU_O2P_new(angles1, angles2))
        elif(chk == 4):
            Overlap2Partof2(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l23 = abs(RESU_O2P_new(angles1, angles2))
        elif(chk == 41):
            Overlap2Partof2(angles1, angles2)
            with torch.no_grad():
                angles2[2] = angles1[2]/divide_by
            l23 = abs(RESU_O2P_new(angles1, angles2))
        else:
            #print('Part of relation satisfied')
            l23 = 0

def check_special(angles1,angles2,status):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1]) 
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    chk = 1000
    if(status == 3):
        if(x1==x2 and y1==y2 and z1==z2):
            if(r1 <= r2):
                chk = 3
                #print('3: First Chrome Part of Second Chrome, all X are Y')
    elif(status == 5):
        if(x1==x2 and y1==y2 and z1==z2):
            if(r2 <= r1):
                chk = 5
                #print('4: Second Chrome Part of First Chrome')
    return chk

@app.route("/animate2")
def animate2():
    global main_count, q1, q2, q3, angles1, angles2, angles3, rel12, rel23, rel13, maxIter, maxRounds
    global lr1, lr2, l12, l23, l13
    global res
    list1 = [res[0]]
    list2 = [res[1]]
    list3 = [res[2]]

    for itr1 in list1:
        for itr2 in list2:
            for itr3 in list3:
                main_count = main_count+1
                #if(main_count<42):
                #    continue
                print("=================",main_count,"=====================\n")
                print("The values are: ", itr1, itr2, itr3)
                
                q1,q2,q3 = create_chromes() 
                angles1 = torch.from_numpy(q1)
                angles1.requires_grad = True
                angles2 = torch.from_numpy(q2)
                angles2.requires_grad = True
                angles3 = torch.from_numpy(q3)
                angles3.requires_grad = True
                print("Chrome1 : ", q1)
                print("Chrome2 : ", q2)
                print("Chrome3 : ", q3)
                table = np.zeros((4, 6))
                table[0][2] = 1
                table[2][4] = 1
                table[0][4] = 1
                
                for i in range(4):
                    for j in range(6):
                        if(table[i][j] != 0):
                            if(i == 0 and j == 2):
                                rel12 = table[i][j]
                            if(i == 1 and j == 2):
                                rel12 = table[i][j]
                            if(i == 0 and j == 3):
                                rel12 = table[i][j]
                            if(i == 1 and j == 3):
                                rel12 = table[i][j]               
                            if(i == 2 and j == 4):
                                rel23 = table[i][j]
                            if(i == 3 and j == 4):
                                rel23 = table[i][j]
                            if(i == 2 and j == 5):
                                rel23 = table[i][j]
                            if(i == 3 and j == 5):
                                rel23 = table[i][j]                
                            if(i == 0 and j == 4):
                                rel13 = table[i][j]
                            if(i == 1 and j == 4):
                                rel13 = table[i][j]
                            if(i == 0 and j == 5):
                                rel13 = table[i][j]
                            if(i == 1 and j == 5):
                                rel13 = table[i][j]
                                
                gLoss = 1
                rounds = 0
                
                

                lr1 = 0.001
                lr2 = 0.01
                # Change r1 and r2 for angles1, angles2 and angles3
                l12 = 1
                l23 = 1
                l13 = 1
                while(rounds < maxRounds and gLoss > 0):     
                    relation = rel12
                    run12(relation)
                # ===============================================================================       
                    relation = rel23
                    run23(relation) # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                # =============================================================================            
                    relation = rel13
                    run13(relation)

                    gLoss = l12+l23+l13
                    rounds = rounds + 1
                loss12(1)
                loss23(1)
                loss13(1)
                true_loss = l12+l23+l13
                print("Actual loss for all disconnected is :", true_loss)
                    
                table[0][2] = itr1
                table[2][4] = itr2
                table[0][4] = itr3
                
                for i in range(4):
                    for j in range(6):
                        if(table[i][j] != 0):
                            if(i == 0 and j == 2):
                                rel12 = table[i][j]
                            if(i == 1 and j == 2):
                                rel12 = table[i][j]
                            if(i == 0 and j == 3):
                                rel12 = table[i][j]
                            if(i == 1 and j == 3):
                                rel12 = table[i][j]               
                            if(i == 2 and j == 4):
                                rel23 = table[i][j]
                            if(i == 3 and j == 4):
                                rel23 = table[i][j]
                            if(i == 2 and j == 5):
                                rel23 = table[i][j]
                            if(i == 3 and j == 5):
                                rel23 = table[i][j]                
                            if(i == 0 and j == 4):
                                rel13 = table[i][j]
                            if(i == 1 and j == 4):
                                rel13 = table[i][j]
                            if(i == 0 and j == 5):
                                rel13 = table[i][j]
                            if(i == 1 and j == 5):
                                rel13 = table[i][j]
                
                gLoss = 1
                gLoss = 1
                rounds = 0

                lr1 = 0.001
                lr2 = 0.01
                # Change r1 and r2 for angles1, angles2 and angles3
                l12 = 1
                l23 = 1
                l13 = 1
                while(rounds < maxRounds and gLoss > 0):     
                    relation = rel12
                    run12(relation)
                # ===============================================================================       
                    relation = rel23
                    run23(relation) # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                # =============================================================================            
                    relation = rel13
                    run13(relation)

                    gLoss = l12+l23+l13
                    rounds = rounds + 1
                    
                    
        # CHANGES MADE HERE
                    
                    
                gLoss = 1
                rounds = 0

                lr1 = 0.001
                lr2 = 0.01
                # Change r1 and r2 for angles1, angles2 and angles3
                l12 = 1
                l23 = 1
                l13 = 1
                while(rounds < maxRounds and gLoss > 0):     
                    relation = rel12
                    run12(relation)
                # ===============================================================================       
                    relation = rel23
                    run23(relation) # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                # =============================================================================            
                    relation = rel13
                    run13(relation)

                    gLoss = l12+l23+l13
                    rounds = rounds + 1
                    
                gLoss = 1
                rounds = 0
                
                lr1 = 0.001
                lr2 = 0.01
                # Change r1 and r2 for angles1, angles2 and angles3
                l12 = 1
                l23 = 1
                l13 = 1
                while(rounds < maxRounds and gLoss > 0):     
                    relation = rel12
                    run12_1fix(relation)
                # ===============================================================================       
                    relation = rel23
                    run23(relation) # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                # =============================================================================            
                    relation = rel13
                    run13(relation)

                    gLoss = l12+l23+l13
                    rounds = rounds + 1  
        # CHANGES END HERE          
                    
                    
                loss12(itr1)
                loss23(itr2)
                loss13(itr3)
                true_loss = l12+l23+l13
                print("Angle1: ", angles1)
                print("Angle2: ", angles2)
                print("Angle3: ", angles3)
                print(check_relation(angles1,angles2),check_relation(angles2,angles3),check_relation(angles1,angles3))
                print("Actual loss for given orientation is :", true_loss)
                print("\n======================================")
                q11,q21,q31 = create_chromes() 
                angles11 = torch.from_numpy(q11)
                angles11.requires_grad = True
                if(itr1==check_special(angles11,angles11,itr1) and itr2==check_special(angles11,angles11,itr2) and itr3==check_special(angles11,angles11,itr3)):
                    print('special case detected and loss is 0')
                print("\n======================================")
    save_1 = []
    save_2 = []
    save_3 = []
    with torch.no_grad():
        tp1 = angles1
        tp2 = angles2
        tp3 = angles3
        np_arr1 = tp1.numpy()
        np_arr2 = tp2.numpy()
        np_arr3 = tp3.numpy()
        ls1 = []
        ls2 = []
        ls3 = []
        ls1 = np_arr1.tolist()
        ls2 = np_arr2.tolist()
        ls3 = np_arr3.tolist()
        save_1.append(ls1)
        save_2.append(ls2)
        save_3.append(ls3)

    df1 = pd.DataFrame(save_1, columns = ['theta1','phi1','alpha1'])
    df2 = pd.DataFrame(save_2, columns = ['theta2','phi2','alpha2'])
    df3 = pd.DataFrame(save_3, columns = ['theta3','phi3','alpha3'])
    df_concat = pd.concat([df1, df2, df3], axis=1)
    #print(df1.head())
    #new_df.columns =['X_axis', 'Y_axis', 'Z_axis', 'Radius']
    json_df_concat = df_concat.to_dict(orient='records')
    #json_df2 = df2.to_dict(orient='records')
    return render_template('animate2.html', title = 'PythonIsHere!', JSON_data = json_df_concat)

