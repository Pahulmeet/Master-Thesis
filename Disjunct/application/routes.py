from application import app
from flask import render_template, url_for, request
import pandas as pd
import json
import plotly
import plotly.express as px
import re
import torch
import torch.nn as nn
import numpy as np
import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

res = []
strval = []
stepval = []
steps = 500
isneg = [0, 0, 0]

wecheck = [0, 0, 0, 0, 0, 0]
wecheck2 = [0, 0, 0]


@app.route('/test', methods=['POST'])
def test():
    global strval, steps, stepval, isneg
    strval.clear()
    isneg.clear()
    output = request.get_json()
    print(output)
    result = json.loads(output)
    global res
    res.clear()
    stepval.clear()

    strval.append(result['state1'])
    strval.append(result['state2'])
    strval.append(result['state3'])
    stepval.append(result['steps'])
    stepval.append(result['steps'])
    stepval.append(result['steps'])

    if "~s" in result["state1"].lower():
        if ("~m" in result["state1"].lower()):
            wecheck2[0] = 53
        else:
            wecheck2[0] = 51
    else:
        if ("~m" in result["state1"].lower()):
            wecheck2[0] = 52
        else:
            wecheck2[0] = 50

    if ("~m" in result["state2"].lower()):
        if ("~p" in result["state2"].lower()):
            wecheck2[1] = 57
        else:
            wecheck2[1] = 55
    else:
        if ("~p" in result["state2"].lower()):
            wecheck2[1] = 56
        else:
            wecheck2[1] = 54

    if ("~s" in result["state3"].lower()):
        if "~p" in result["state3"].lower():
            wecheck2[2] = 61
        else:
            wecheck2[2] = 59
    else:
        if ("~p" in result["state3"].lower()):
            wecheck2[2] = 60
        else:
            wecheck2[2] = 58

    print("wecheck is : ", wecheck2)

    for i in result:
        result[i] = result[i].strip()
        result[i] = " ".join(result[i].split())
        if "~s" in result[i].lower():
            isneg.append(1)  # 1 is S
        if "~m" in result[i].lower():
            isneg.append(2)  # 2 is M
        if "~p" in result[i].lower():
            isneg.append(3)  # 3 is P

        result[i] = result[i].replace("~", "")
    print("result is : ", result)

    if (result['state1'].casefold() == "NO S ARE M".casefold() or result['state1'].casefold() == 'NO M ARE S'.casefold()):
        res.append(1)
    elif (result['state1'].casefold() == 'SOME S ARE M'.casefold() or result['state1'].casefold() == 'SOME M ARE S'.casefold()):
        res.append(2)
    elif (result['state1'].casefold() == 'ALL S ARE M'.casefold()):
        res.append(3)
    elif (result['state1'].casefold() == 'ALL M ARE S'.casefold()):
        res.append(5)
    elif (result['state1'].casefold() == 'SOME S ARE NOT M'.casefold()):
        res.append(4)
    elif (result['state1'].casefold() == 'SOME M ARE NOT S'.casefold()):
        res.append(41)

    if (result['state2'].casefold() == "NO M ARE P".casefold() or result['state2'].casefold() == 'NO P ARE M'.casefold()):
        res.append(1)
    elif (result['state2'].casefold() == 'SOME M ARE P'.casefold() or result['state2'].casefold() == 'SOME P ARE M'.casefold()):
        res.append(2)
    elif (result['state2'].casefold() == 'ALL M ARE P'.casefold()):
        res.append(3)
    elif (result['state2'].casefold() == 'ALL P ARE M'.casefold()):
        res.append(5)
    elif (result['state2'].casefold() == 'SOME M ARE NOT P'.casefold()):
        res.append(4)
    elif (result['state2'].casefold() == 'SOME P ARE NOT M'.casefold()):
        res.append(41)

    if (result['state3'].casefold() == "NO S ARE P".casefold() or result['state3'].casefold() == 'NO P ARE S'.casefold()):
        res.append(1)
    elif (result['state3'].casefold() == 'SOME S ARE P'.casefold() or result['state3'].casefold() == 'SOME P ARE S'.casefold()):
        res.append(2)
    elif (result['state3'].casefold() == 'ALL S ARE P'.casefold()):
        res.append(3)
    elif (result['state3'].casefold() == 'ALL P ARE S'.casefold()):
        res.append(5)
    elif (result['state3'].casefold() == 'SOME S ARE NOT P'.casefold()):
        res.append(4)
    elif (result['state3'].casefold() == 'SOME P ARE NOT S'.casefold()):
        res.append(41)
    steps = result['steps']
    print(steps)

    return result


@app.route('/help')
def help():
    return render_template('help.html', title='PythonIsHere!')


rho = 1  # Radius of the Sphere
theta = 0  # Angle on the X-Y Plane between [0,2*pi]
phi = 0  # ANgle on the Z-Axis between [0, pi]

main_count = 0
true_loss = 100
q1 = np.array([])
q2 = np.array([])
q3 = np.array([])
angles1 = torch.tensor([])
angles2 = torch.tensor([])
angles3 = torch.tensor([])
angles1c = torch.tensor([])
angles2c = torch.tensor([])
angles3c = torch.tensor([])
rel12 = 0
rel23 = 0
rel13 = 0
maxIter = 4500  # was 3000
maxRounds = 1

lr1 = 0.001  # 0.001
lr2 = 0.01  # 0.01

l12 = 1
l23 = 1
l13 = 1

C1 = []
C2 = []
C3 = []


def distance_between_centres(c1, c2):
    d = rho*np.arccos((c1[0].item()*c2[0].item() + c1[1].item()
                      * c2[1].item() + c1[2].item()*c2[2].item())/(rho*rho))
    return d


def distance_between_points(c1, dot_x, dot_y, dot_z):
    d = rho*np.arccos((c1[0].item()*dot_x + c1[1].item()
                      * dot_y + c1[2].item()*dot_z)/(rho*rho))
    return d


@app.route("/")
def index():
    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    chrome_angle = 110

    random_alpha = np.array([chrome_angle])
    global q1, r1
    q1 = np.append(random_theta, random_phi)
    q1 = np.append(q1, random_alpha)
    x1 = rho*np.sin(random_phi)*np.cos(random_theta)
    y1 = rho*np.sin(random_phi)*np.sin(random_theta)
    z1 = rho*np.cos(random_phi)
    r1 = rho*np.pi*(random_alpha/360)  # Radius of Chrome
    l1 = [x1, y1, z1, r1]
    c1 = torch.tensor(l1, dtype=torch.float32, requires_grad=True)

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi

    global q2, r2
    q2 = np.append(random_theta, random_phi)
    q2 = np.append(q2, random_alpha)
    x2 = rho*np.sin(random_phi)*np.cos(random_theta)
    y2 = rho*np.sin(random_phi)*np.sin(random_theta)
    z2 = rho*np.cos(random_phi)
    r2 = rho*np.pi*(random_alpha/360)  # Radius of Chrome
    l2 = [x2, y2, z2, r2]
    c2 = torch.tensor(l2, dtype=torch.float32, requires_grad=True)

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi

    global q3, r3
    q3 = np.append(random_theta, random_phi)
    q3 = np.append(q3, random_alpha)
    x3 = rho*np.sin(random_phi)*np.cos(random_theta)
    y3 = rho*np.sin(random_phi)*np.sin(random_theta)
    z3 = rho*np.cos(random_phi)
    r3 = rho*np.pi*(random_alpha/360)  # Radius of Chrome
    l3 = [x3, y3, z3, r3]
    c3 = torch.tensor(l3, dtype=torch.float32, requires_grad=True)

    temp_df1 = pd.DataFrame(l1)
    temp_df1 = temp_df1.T
    temp_df2 = pd.DataFrame(l2)
    temp_df2 = temp_df2.T
    temp_df3 = pd.DataFrame(l3)
    temp_df3 = temp_df3.T
    df = temp_df1.append(temp_df2)
    df = df.append(temp_df3)
    df.columns = ['X_axis', 'Y_axis', 'Z_axis', 'Radius']

    global json_data
    json_data = df.to_dict(orient='records')

    return render_template('index.html', title='Syllogisms', JSON_data=json_data)


def distance_between_centres(c1, c2):
    d = rho*np.arccos((c1[0].item()*c2[0].item() + c1[1].item()
                      * c2[1].item() + c1[2].item()*c2[2].item())/(rho*rho))
    return d


def check_disconnect(d, c1, c2):
    chk = 1000
    if (d - (c1[3].item() + c2[3].item()) >= 0):
        chk = 1
    else:
        chk = 0
    return chk


def check_overlap(d, c1, c2):
    chk = 1000
    if (d > abs(c1[3].item() - c2[3].item()) and d < abs(c1[3].item() + c2[3].item())):
        chk = 1
    else:
        chk = 0
    return chk


def check_part_of(d, c1, c2):
    chk = 1000
    if (d + c1[3].item() <= c2[3].item()):
        chk = 1
    else:
        chk = 0
    return chk


tolerance1 = 0.14  # originally was 0.01\n",
tolerance2 = 0.1  # originally was 0.03\n",

# origunally was 0.02 (should be slightly more than tolerance1 in training function)\n",
tolerance3 = 0.15
divide_by = 5.5


def check_relation(angles1, angles2):
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
    if (d.item() - (r1 + r2) >= 0):
        chk = 1
        # print('1: Disconnect, no X are Y')
    elif (d.item() + r1 <= r2):
        chk = 3
        # print('3: First Chrome Part of Second Chrome, all X are Y')
    elif (d.item() + r2 <= r1):
        chk = 5
        # print('4: Second Chrome Part of First Chrome')
    elif (d.item() - (r1 + r2) < 0):
        chk = 2
        # print('2: Partial Overlap, some X are Y')
    elif (d.item() + r1 > r2):
        chk = 4
    elif (d.item() + r2 > r1):
        chk = 41
        # print('4: Second Chrome Part of First Chrome, some Y are not X')
    return chk


def distance_between_points(c1, dot_x, dot_y, dot_z):
    d = rho*np.arccos((c1[0].item()*dot_x + c1[1].item()
                      * dot_y + c1[2].item()*dot_z)/(rho*rho))
    return d


def RESU_D2O_new(angles1, angles2):
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


def RESU_O2D_new(angles1, angles2):
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


def RESU_O2P_new(angles1, angles2):
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


def RESU_P2O_new(angles1, angles2):
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


def RESU_O2IP_new(angles1, angles2):
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


def RESU_IP2O_new(angles1, angles2):
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


def print_all(angles1, angles2):
    x1 = rho*torch.sin(angles1[1])*torch.cos(angles1[0])
    y1 = rho*torch.sin(angles1[1])*torch.sin(angles1[0])
    z1 = rho*torch.cos(angles1[1])
    x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
    y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
    z2 = rho*torch.cos(angles2[1])
    d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
    r1 = rho*np.pi*(angles1[2].item()/360)
    r2 = rho*np.pi*(angles2[2].item()/360)
    print("Co-ordinates of C1:", x1.item(), y1.item(), z1.item())
    print("Co-ordinates of C2:", x2.item(), y2.item(), z2.item())
    print("Radius of C1:", r1)
    print("Radius of C2:", r2)
    print("Sum of Radii:", r1+r2, " and Distance between them:", d.item())


rho = 1  # Radius of the Sphere
theta = 0  # Angle on the X-Y Plane between [0,2*pi]
phi = 0  # Angle on the Z-Axis between [0, pi]
chrome_angle = 110


def create_chromes():
    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    # random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([chrome_angle])
    q1 = np.append(random_theta, random_phi)
    q1 = np.append(q1, random_alpha)
    x1 = rho*np.sin(random_phi)*np.cos(random_theta)
    y1 = rho*np.sin(random_phi)*np.sin(random_theta)
    z1 = rho*np.cos(random_phi)
    r1 = rho*np.pi*(random_alpha/360)  # Radius of Chrome
    l1 = [x1, y1, z1, r1]
    c1 = torch.tensor(l1, dtype=torch.float32, requires_grad=True)
    # print("Chrome 1 Co-ordinates and radius (x,y,z,r) : ", "x:", c1[0].item()," y:",c1[1].item()," z:",c1[2].item()," r:",c1[3].item())

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    # random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([chrome_angle])
    q2 = np.append(random_theta, random_phi)
    q2 = np.append(q2, random_alpha)
    x2 = rho*np.sin(random_phi)*np.cos(random_theta)
    y2 = rho*np.sin(random_phi)*np.sin(random_theta)
    z2 = rho*np.cos(random_phi)
    r2 = rho*np.pi*(random_alpha/360)  # Radius of Chrome
    l2 = [x2, y2, z2, r2]
    c2 = torch.tensor(l2, dtype=torch.float32, requires_grad=True)
    # print("Chrome 2 Co-ordinates and radius (x,y,z,r) : ", "x:", c2[0].item()," y:",c2[1].item()," z:",c2[2].item()," r:",c2[3].item())

    random_theta = (0 - 2*np.pi) * np.random.rand(1) + 2*np.pi
    random_phi = (0 - np.pi) * np.random.rand(1) + np.pi
    # random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
    random_alpha = np.array([chrome_angle])
    q3 = np.append(random_theta, random_phi)
    q3 = np.append(q3, random_alpha)
    x3 = rho*np.sin(random_phi)*np.cos(random_theta)
    y3 = rho*np.sin(random_phi)*np.sin(random_theta)
    z3 = rho*np.cos(random_phi)
    r3 = rho*np.pi*(random_alpha/360)  # Radius of Chrome
    l3 = [x3, y3, z3, r3]
    c3 = torch.tensor(l3, dtype=torch.float32, requires_grad=True)
    # print("Chrome 3 Co-ordinates and radius (x,y,z,r) : ", "x:", c3[0].item()," y:",c3[1].item()," z:",c3[2].item()," r:",c3[3].item())
    return q1, q2, q3


def create_complement(angle):
    thetaC = 0.0
    phiC = 0.0
    alphaC = 0.0
    with torch.no_grad():
        thetaC = np.pi + angle[0]
        if (thetaC > 2*np.pi):
            thetaC = thetaC-2*np.pi
        phiC = np.pi - angle[1]
        # random_alpha = np.array([np.random.randint(10,90)]) # Angle the chrome makes with the centre of the Sphere
        alphaC = 360 - angle[2] - 25
    q1c = np.array([thetaC, phiC, alphaC])
    return q1c


def run12(relation):
    global l12
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    if (wecheck2[0] == 50):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect(angles1, angles2)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1, angles2)
                ts11, ts22 = Overlap2Disconnect(angles1, angles2)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1, angles2)
                ts11, ts22 = Overlap2Disconnect(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1, angles2))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1, angles2)
                ts11, ts22 = Overlap2InversePartof(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1, angles2)
                ts11, ts22 = Overlap2InversePartof(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1, angles2)
                ts11, ts22 = Overlap2Partof(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1, angles2)
                ts11, ts22 = Overlap2Partof(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0

    elif (wecheck2[0] == 51):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect(angles1c, angles2)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1c, angles2)
                ts11, ts22 = Overlap2Disconnect(angles1c, angles2)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1c, angles2)
                ts11, ts22 = Overlap2Disconnect(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1c, angles2))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1c, angles2)
                ts11, ts22 = Overlap2InversePartof(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1c, angles2)
                ts11, ts22 = Overlap2InversePartof(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1c, angles2)
                ts11, ts22 = Overlap2Partof(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1c, angles2)
                ts11, ts22 = Overlap2Partof(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0
        q1c = create_complement(angles1c)
        angles1 = torch.from_numpy(q1c)
        angles1.requires_grad = True

    elif (wecheck2[0] == 52):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect(angles1, angles2c)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1, angles2c)
                ts11, ts22 = Overlap2Disconnect(angles1, angles2c)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1, angles2c)
                ts11, ts22 = Overlap2Disconnect(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1, angles2c))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1, angles2c)
                ts11, ts22 = Overlap2InversePartof(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1, angles2c)
                ts11, ts22 = Overlap2InversePartof(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1, angles2c)
                ts11, ts22 = Overlap2Partof(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1, angles2c)
                ts11, ts22 = Overlap2Partof(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            else:
                # print('Part of relation satisfied')
                l12 = 0
        q1c = create_complement(angles2c)
        angles2 = torch.from_numpy(q1c)
        angles2.requires_grad = True

    elif (wecheck2[0] == 53):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect(angles1c, angles2c)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1c, angles2c)
                ts11, ts22 = Overlap2Disconnect(angles1c, angles2c)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1c, angles2c)
                ts11, ts22 = Overlap2Disconnect(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1c, angles2c)
                ts11, ts22 = Overlap2InversePartof(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1c, angles2c)
                ts11, ts22 = Overlap2InversePartof(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap(angles1c, angles2c)
                ts11, ts22 = Overlap2Partof(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap(angles1c, angles2c)
                ts11, ts22 = Overlap2Partof(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0
        q1c = create_complement(angles1c)
        angles1 = torch.from_numpy(q1c)
        angles1.requires_grad = True
        q1c = create_complement(angles2c)
        angles2 = torch.from_numpy(q1c)
        angles2.requires_grad = True


def run23(relation):
    global l23
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    if (wecheck2[1] == 54):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 2):
                ts2, ts3 = Overlap2Disconnect2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2, angles3)
                ts22, ts33 = Overlap2Disconnect2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2Disconnect2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 41):
                ts2, ts3 = Overlap2Disconnect2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 5):
                Partof2Overlap2(angles2, angles3)
                Overlap2Disconnect2(angles2, angles3)
                l23 = abs(RESU_O2D_new(angles2, angles3))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_D2O_new(angles2, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2, angles3)
                ts22, ts33 = Overlap2InversePartof2(angles2, angles3)
                with torch.no_grad():
                    angles2[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 2):
                ts2, ts3 = Overlap2InversePartof2(angles2, angles3)
                with torch.no_grad():
                    angles2[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2InversePartof2(angles2, angles3)
                with torch.no_grad():
                    angles2[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 41):
                ts2, ts3 = Overlap2InversePartof2(angles2, angles3)
                with torch.no_grad():
                    angles2[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2, angles3)
                ts22, ts33 = Overlap2InversePartof2(angles2, angles3)
                with torch.no_grad():
                    angles2[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_IP2O_new(angles2, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_P2O_new(angles2, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2, angles3)
                ts22, ts33 = Overlap2Partof2(angles2, angles3)
                with torch.no_grad():
                    angles3[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 2):
                ts2, ts3 = Overlap2Partof2(angles2, angles3)
                with torch.no_grad():
                    angles3[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2, angles3)
                ts22, ts33 = Overlap2Partof2(angles2, angles3)
                with torch.no_grad():
                    angles3[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2, angles3)
                with torch.no_grad():
                    angles3[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2, angles3)
                with torch.no_grad():
                    angles3[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2, angles3))
            else:
                # print('Part of relation satisfied')
                l23 = 0

    elif (wecheck2[1] == 55):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 2):
                ts2, ts3 = Overlap2Disconnect2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2c, angles3)
                ts22, ts33 = Overlap2Disconnect2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2Disconnect2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 41):
                ts2, ts3 = Overlap2Disconnect2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 5):
                Partof2Overlap2(angles2c, angles3)
                Overlap2Disconnect2(angles2c, angles3)
                l23 = abs(RESU_O2D_new(angles2c, angles3))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_D2O_new(angles2c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2c, angles3)
                ts22, ts33 = Overlap2InversePartof2(angles2c, angles3)
                with torch.no_grad():
                    angles2c[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 2):
                ts2, ts3 = Overlap2InversePartof2(angles2c, angles3)
                with torch.no_grad():
                    angles2c[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2InversePartof2(angles2c, angles3)
                with torch.no_grad():
                    angles2c[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 41):
                ts2, ts3 = Overlap2InversePartof2(angles2c, angles3)
                with torch.no_grad():
                    angles2c[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2c, angles3)
                ts22, ts33 = Overlap2InversePartof2(angles2c, angles3)
                with torch.no_grad():
                    angles2c[2] = angles3[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_IP2O_new(angles2c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2c, angles3)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_P2O_new(angles2c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2c, angles3)
                ts22, ts33 = Overlap2Partof2(angles2c, angles3)
                with torch.no_grad():
                    angles3[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 2):
                ts2, ts3 = Overlap2Partof2(angles2c, angles3)
                with torch.no_grad():
                    angles3[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2c, angles3)
                ts22, ts33 = Overlap2Partof2(angles2c, angles3)
                with torch.no_grad():
                    angles3[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2c, angles3)
                with torch.no_grad():
                    angles3[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2c, angles3)
                with torch.no_grad():
                    angles3[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            else:
                # print('Part of relation satisfied')
                l23 = 0

        q1c = create_complement(angles2c)
        angles2 = torch.from_numpy(q1c)
        angles2.requires_grad = True

    if (wecheck2[1] == 56):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 2):
                ts2, ts3 = Overlap2Disconnect2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2, angles3c)
                ts22, ts33 = Overlap2Disconnect2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2Disconnect2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 41):
                ts2, ts3 = Overlap2Disconnect2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 5):
                Partof2Overlap2(angles2, angles3c)
                Overlap2Disconnect2(angles2, angles3c)
                l23 = abs(RESU_O2D_new(angles2, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_D2O_new(angles2, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2, angles3c)
                ts22, ts33 = Overlap2InversePartof2(angles2, angles3c)
                with torch.no_grad():
                    angles2[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 2):
                ts2, ts3 = Overlap2InversePartof2(angles2, angles3c)
                with torch.no_grad():
                    angles2[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2InversePartof2(angles2, angles3c)
                with torch.no_grad():
                    angles2[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 41):
                ts2, ts3 = Overlap2InversePartof2(angles2, angles3c)
                with torch.no_grad():
                    angles2[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2, angles3c)
                ts22, ts33 = Overlap2InversePartof2(angles2, angles3c)
                with torch.no_grad():
                    angles2[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_IP2O_new(angles2, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_P2O_new(angles2, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2, angles3c)
                ts22, ts33 = Overlap2Partof2(angles2, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 2):
                ts2, ts3 = Overlap2Partof2(angles2, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2, angles3c)
                ts22, ts33 = Overlap2Partof2(angles2, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            else:
                # print('Part of relation satisfied')
                l23 = 0
        q1c = create_complement(angles3c)
        angles3 = torch.from_numpy(q1c)
        angles3.requires_grad = True

    if (wecheck2[1] == 57):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 2):
                ts2, ts3 = Overlap2Disconnect2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2c, angles3c)
                ts22, ts33 = Overlap2Disconnect2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2Disconnect2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 41):
                ts2, ts3 = Overlap2Disconnect2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 5):
                Partof2Overlap2(angles2c, angles3c)
                Overlap2Disconnect2(angles2c, angles3c)
                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_D2O_new(angles2c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2c, angles3c)
                ts22, ts33 = Overlap2InversePartof2(angles2c, angles3c)
                with torch.no_grad():
                    angles2c[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 2):
                ts2, ts3 = Overlap2InversePartof2(angles2c, angles3c)
                with torch.no_grad():
                    angles2c[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2InversePartof2(angles2c, angles3c)
                with torch.no_grad():
                    angles2c[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 41):
                ts2, ts3 = Overlap2InversePartof2(angles2c, angles3c)
                with torch.no_grad():
                    angles2c[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2c, angles3c)
                ts22, ts33 = Overlap2InversePartof2(angles2c, angles3c)
                with torch.no_grad():
                    angles2c[2] = angles3c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_IP2O_new(angles2c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 5):
                ts2, ts3 = Partof2Overlap2(angles2c, angles3c)
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_P2O_new(angles2c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):
                ts2, ts3 = Disconnect2Overlap2(angles2c, angles3c)
                ts22, ts33 = Overlap2Partof2(angles2c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 2):
                ts2, ts3 = Overlap2Partof2(angles2c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 3):
                ts2, ts3 = InversePartof2Overlap2(angles2c, angles3c)
                ts22, ts33 = Overlap2Partof2(angles2c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)
                C2.append(ts22)
                C3.append(ts33)

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 4):
                ts2, ts3 = Overlap2Partof2(angles2c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles2c[2]/divide_by
                C2.append(ts2)
                C3.append(ts3)

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            else:
                # print('Part of relation satisfied')
                l23 = 0
        q1c = create_complement(angles2c)
        angles2 = torch.from_numpy(q1c)
        angles2.requires_grad = True
        q1c = create_complement(angles3c)
        angles3 = torch.from_numpy(q1c)
        angles3.requires_grad = True


def run13(relation):
    global l13
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    temp_relation = rel23
    run23(temp_relation)  # Problem, correct relation
    if (wecheck2[2] == 58):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 2):
                ts1, ts3 = Overlap2Disconnect(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1, angles3)
                ts11, ts33 = Overlap2Disconnect(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 4):
                ts1, ts3 = Overlap2Disconnect(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 41):
                ts1, ts3 = Overlap2Disconnect(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1, angles3)
                ts11, ts33 = Overlap2Disconnect(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1, angles3))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_D2O_new(angles1, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1, angles3)
                ts11, ts33 = Overlap2InversePartof(angles1, angles3)
                with torch.no_grad():
                    angles1[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 2):
                ts1, ts3 = Overlap2InversePartof(angles1, angles3)
                with torch.no_grad():
                    angles1[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 4):
                ts1, ts3 = Overlap2InversePartof(angles1, angles3)
                with torch.no_grad():
                    angles1[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 41):
                ts1, ts3 = Overlap2InversePartof(angles1, angles3)
                with torch.no_grad():
                    angles1[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1, angles3)
                ts11, ts33 = Overlap2InversePartof(angles1, angles3)
                with torch.no_grad():
                    angles1[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_IP2O_new(angles1, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_P2O_new(angles1, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1, angles3)
                ts11, ts33 = Overlap2Partof(angles1, angles3)
                with torch.no_grad():
                    angles3[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 2):
                ts1, ts3 = Overlap2Partof(angles1, angles3)
                with torch.no_grad():
                    angles3[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1, angles3)
                ts11, ts33 = Overlap2Partof(angles1, angles3)
                with torch.no_grad():
                    angles3[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 4):
                ts1, ts3 = Overlap2Partof(angles1, angles3)
                with torch.no_grad():
                    angles3[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 41):
                ts1, ts3 = Overlap2Partof(angles1, angles3)
                with torch.no_grad():
                    angles3[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1, angles3))
            else:
                # print('Part of relation satisfied')
                l13 = 0

    elif (wecheck2[2] == 59):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 2):
                ts1, ts3 = Overlap2Disconnect(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1c, angles3)
                ts11, ts33 = Overlap2Disconnect(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 4):
                ts1, ts3 = Overlap2Disconnect(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 41):
                ts1, ts3 = Overlap2Disconnect(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1c, angles3)
                ts11, ts33 = Overlap2Disconnect(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_D2O_new(angles1c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1c, angles3)
                ts11, ts33 = Overlap2InversePartof(angles1c, angles3)
                with torch.no_grad():
                    angles1c[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 2):
                ts1, ts3 = Overlap2InversePartof(angles1c, angles3)
                with torch.no_grad():
                    angles1c[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 4):
                ts1, ts3 = Overlap2InversePartof(angles1c, angles3)
                with torch.no_grad():
                    angles1c[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 41):
                ts1, ts3 = Overlap2InversePartof(angles1c, angles3)
                with torch.no_grad():
                    angles1c[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1c, angles3)
                ts11, ts33 = Overlap2InversePartof(angles1c, angles3)
                with torch.no_grad():
                    angles1c[2] = angles3[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_IP2O_new(angles1c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1c, angles3)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_P2O_new(angles1c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1c, angles3)
                ts11, ts33 = Overlap2Partof(angles1c, angles3)
                with torch.no_grad():
                    angles3[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 2):
                ts1, ts3 = Overlap2Partof(angles1c, angles3)
                with torch.no_grad():
                    angles3[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1c, angles3)
                ts11, ts33 = Overlap2Partof(angles1c, angles3)
                with torch.no_grad():
                    angles3[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 4):
                ts1, ts3 = Overlap2Partof(angles1c, angles3)
                with torch.no_grad():
                    angles3[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 41):
                ts1, ts3 = Overlap2Partof(angles1c, angles3)
                with torch.no_grad():
                    angles3[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            else:
                # print('Part of relation satisfied')
                l13 = 0
        q1c = create_complement(angles1c)
        angles1 = torch.from_numpy(q1c)
        angles1.requires_grad = True

    if (wecheck2[2] == 60):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 2):
                ts1, ts3 = Overlap2Disconnect(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1, angles3c)
                ts11, ts33 = Overlap2Disconnect(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 4):
                ts1, ts3 = Overlap2Disconnect(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 41):
                ts1, ts3 = Overlap2Disconnect(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1, angles3c)
                ts11, ts33 = Overlap2Disconnect(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_D2O_new(angles1, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1, angles3c)
                ts11, ts33 = Overlap2InversePartof(angles1, angles3c)
                with torch.no_grad():
                    angles1[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 2):
                ts1, ts3 = Overlap2InversePartof(angles1, angles3c)
                with torch.no_grad():
                    angles1[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 4):
                ts1, ts3 = Overlap2InversePartof(angles1, angles3c)
                with torch.no_grad():
                    angles1[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 41):
                ts1, ts3 = Overlap2InversePartof(angles1, angles3c)
                with torch.no_grad():
                    angles1[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1, angles3c)
                ts11, ts33 = Overlap2InversePartof(angles1, angles3c)
                with torch.no_grad():
                    angles1[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_IP2O_new(angles1, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_P2O_new(angles1, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1, angles3c)
                ts11, ts33 = Overlap2Partof(angles1, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 2):
                ts1, ts3 = Overlap2Partof(angles1, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1, angles3c)
                ts11, ts33 = Overlap2Partof(angles1, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 4):
                ts1, ts3 = Overlap2Partof(angles1, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 41):
                ts1, ts3 = Overlap2Partof(angles1, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            else:
                # print('Part of relation satisfied')
                l13 = 0
        q1c = create_complement(angles3c)
        angles3 = torch.from_numpy(q1c)
        angles3.requires_grad = True

    if (wecheck2[2] == 61):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 2):
                ts1, ts3 = Overlap2Disconnect(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1c, angles3c)
                ts11, ts33 = Overlap2Disconnect(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 4):
                ts1, ts3 = Overlap2Disconnect(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 41):
                ts1, ts3 = Overlap2Disconnect(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1c, angles3c)
                ts11, ts33 = Overlap2Disconnect(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_D2O_new(angles1c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1c, angles3c)
                ts11, ts33 = Overlap2InversePartof(angles1c, angles3c)
                with torch.no_grad():
                    angles1c[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 2):
                ts1, ts3 = Overlap2InversePartof(angles1c, angles3c)
                with torch.no_grad():
                    angles1c[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 4):
                ts1, ts3 = Overlap2InversePartof(angles1c, angles3c)
                with torch.no_grad():
                    angles1c[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 41):
                ts1, ts3 = Overlap2InversePartof(angles1c, angles3c)
                with torch.no_grad():
                    angles1c[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1c, angles3c)
                ts11, ts33 = Overlap2InversePartof(angles1c, angles3c)
                with torch.no_grad():
                    angles1c[2] = angles3c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_IP2O_new(angles1c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 5):
                ts1, ts3 = Partof2Overlap(angles1c, angles3c)
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_P2O_new(angles1c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):
                ts1, ts3 = Disconnect2Overlap(angles1c, angles3c)
                ts11, ts33 = Overlap2Partof(angles1c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 2):
                ts1, ts3 = Overlap2Partof(angles1c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 3):
                ts1, ts3 = InversePartof2Overlap(angles1c, angles3c)
                ts11, ts33 = Overlap2Partof(angles1c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)
                C1.append(ts11)
                C3.append(ts33)

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 4):
                ts1, ts3 = Overlap2Partof(angles1c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 41):
                ts1, ts3 = Overlap2Partof(angles1c, angles3c)
                with torch.no_grad():
                    angles3c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C3.append(ts3)

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            else:
                # print('Part of relation satisfied')
                l13 = 0
        q1c = create_complement(angles1c)
        angles1 = torch.from_numpy(q1c)
        angles1.requires_grad = True
        q1c = create_complement(angles3c)
        angles3 = torch.from_numpy(q1c)
        angles3.requires_grad = True


def Overlap2Disconnect(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2
    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    z = 0
    while (z <= maxIter and RESU_O2D_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            # angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        # angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2D_new(angles1,angles2))  """
    return save_1, save_2


def Overlap2Disconnect2(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2
    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    z = 0
    while (z <= maxIter and RESU_O2D_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            # angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        # angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2D_new(angles1,angles2))   """
    return save_1, save_2


def InversePartof2Overlap(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_IP2O_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            # angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        # angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_IP2O_new(angles1,angles2))"""
    return save_1, save_2


def InversePartof2Overlap2(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_IP2O_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            # angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        # angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_IP2O_new(angles1,angles2))"""
    return save_1, save_2


def Overlap2InversePartof(angles1, angles2):
    global mastercount
    lr = lr2

    def loss_part_of(r1, d, r2):
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
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_O2IP_new(angles1, angles2)):
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
        d, r1, r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1, d, r2)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            # angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        # angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2IP_new(angles1,angles2))"""
    return save_1, save_2


def Overlap2InversePartof2(angles1, angles2):
    global mastercount
    lr = lr2

    def loss_part_of(r1, d, r2):
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
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_O2IP_new(angles1, angles2)):
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
        d, r1, r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1, d, r2)
        l.backward()
        with torch.no_grad():
            # angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        # angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2IP_new(angles1,angles2))"""

    return save_1, save_2


def Partof2Overlap(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_P2O_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            # angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        # angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_P2O_new(angles1,angles2))"""

    return save_1, save_2


def Partof2Overlap2(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_P2O_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            # angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        # angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_P2O_new(angles1,angles2))"""
    return save_1, save_2


def Overlap2Partof(angles1, angles2):
    global mastercount
    lr = lr2

    def loss_part_of(r1, d, r2):
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
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    z = 0
    while (z <= maxIter and RESU_O2P_new(angles1, angles2)):
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
        d, r1, r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1, d, r2)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            # angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        # angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2P_new(angles1,angles2))"""

    return save_1, save_2


def Overlap2Partof2(angles1, angles2):
    global mastercount
    lr = lr2

    def loss_part_of(r1, d, r2):
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
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1
    z = 0
    while (z <= maxIter and RESU_O2P_new(angles1, angles2)):
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
        d, r1, r2 = forward1()
        d = d.type(torch.FloatTensor)
        l = loss_part_of(r1, d, r2)
        l.backward()
        with torch.no_grad():
            # angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        # angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_O2P_new(angles1,angles2))"""
    return save_1, save_2


def Disconnect2Overlap(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_D2O_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            angles1 -= lr*angles1.grad
            # angles2 -= lr*angles2.grad
        angles1.grad.zero_()
        # angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_D2O_new(angles1,angles2))"""

    return save_1, save_2


def Disconnect2Overlap2(angles1, angles2):
    global mastercount
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
        r1 = rho*3.141*(angles1[2]/360)
        x2 = rho*torch.sin(angles2[1])*torch.cos(angles2[0])
        y2 = rho*torch.sin(angles2[1])*torch.sin(angles2[0])
        z2 = rho*torch.cos(angles2[1])
        r2 = rho*3.141*(angles2[2]/360)
        d = rho*torch.acos((x1*x2+y1*y2+z1*z2)/(rho*rho))
        return d, r1, r2

    # print(angles1)
    # print(angles2)
    save_1 = []
    save_2 = []
    maincheck = 1

    z = 0
    while (z <= maxIter and RESU_D2O_new(angles1, angles2)):
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
        d, r1, r2 = forward()
        d = d.type(torch.FloatTensor)
        tr = tr.type(torch.FloatTensor)
        l = loss(d, tr)
        l.backward()
        with torch.no_grad():
            # angles1 -= lr*angles1.grad
            angles2 -= lr*angles2.grad
        # angles1.grad.zero_()
        angles2.grad.zero_()
        z = z+1
        mastercount = mastercount+1
        """if(z%20000==0):
            print("Value of d: ", d, " when z is : ", z)
            print("Value of loss : ", l)
            print("Rounds??", rounds, " and ",RESU_D2O_new(angles1,angles2))"""

    return save_1, save_2


def loss12(relation):
    global l12
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    if (wecheck2[0] == 50):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)

            if (chk != relation):
                l12 = 1
            if (chk == 2):

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 3):
                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 4):

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 41):

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 5):

                l12 = abs(RESU_O2D_new(angles1, angles2))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_D2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 2):

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 4):

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 41):

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 5):

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                l12 = abs(RESU_IP2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                l12 = abs(RESU_P2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 2):

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 3):

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 4):

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 41):

                l12 = abs(RESU_O2P_new(angles1, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0

    elif (wecheck2[0] == 51):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)

            if (chk != relation):
                l12 = 1
            if (chk == 2):

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 3):

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 4):

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 41):

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 5):

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_D2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 2):

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 4):

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 41):

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 5):

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                l12 = abs(RESU_IP2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                l12 = abs(RESU_P2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 2):

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 3):

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 4):

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 41):

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0

    elif (wecheck2[0] == 52):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)

            if (chk != relation):
                l12 = 1
            if (chk == 2):

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 3):

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 4):

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 41):

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 5):

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_D2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 2):

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 4):

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 41):

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 5):

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                l12 = abs(RESU_IP2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                l12 = abs(RESU_P2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 2):

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 3):

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 4):

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 41):

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            else:
                # print('Part of relation satisfied')
                l12 = 0

    elif (wecheck2[0] == 53):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)

            if (chk != relation):
                l12 = 1
            if (chk == 2):

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 3):

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 4):

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 41):

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 5):

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_D2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 2):

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 4):

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 41):

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 5):

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                l12 = abs(RESU_IP2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                l12 = abs(RESU_P2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 2):

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 3):

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 4):
                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 41):
                l12 = abs(RESU_O2P_new(angles1c, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0


def loss23(relation):
    global l23
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    if (wecheck2[1] == 54):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 2):

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 3):

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 41):

                l23 = abs(RESU_O2D_new(angles2, angles3))
            elif (chk == 5):

                l23 = abs(RESU_O2D_new(angles2, angles3))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_D2O_new(angles2, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 2):

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 41):

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            elif (chk == 5):

                l23 = abs(RESU_O2IP_new(angles2, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 3):

                l23 = abs(RESU_IP2O_new(angles2, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 5):

                l23 = abs(RESU_P2O_new(angles2, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 2):

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 3):

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2, angles3))
            else:
                # print('Part of relation satisfied')
                l23 = 0

    elif (wecheck2[1] == 55):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 2):

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 3):

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 41):

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            elif (chk == 5):

                l23 = abs(RESU_O2D_new(angles2c, angles3))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_D2O_new(angles2c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 2):

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 41):

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            elif (chk == 5):

                l23 = abs(RESU_O2IP_new(angles2c, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 3):

                l23 = abs(RESU_IP2O_new(angles2c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 5):

                l23 = abs(RESU_P2O_new(angles2c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 2):

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 3):

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2c, angles3))
            else:
                # print('Part of relation satisfied')
                l23 = 0

    if (wecheck2[1] == 56):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 2):

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 3):

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 41):

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            elif (chk == 5):

                l23 = abs(RESU_O2D_new(angles2, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_D2O_new(angles2, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 2):

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 41):

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            elif (chk == 5):

                l23 = abs(RESU_O2IP_new(angles2, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 3):

                l23 = abs(RESU_IP2O_new(angles2, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 5):

                l23 = abs(RESU_P2O_new(angles2, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 2):

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 3):

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2, angles3c))
            else:
                # print('Part of relation satisfied')
                l23 = 0

    if (wecheck2[1] == 57):
        if (relation == 1):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 2):

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 3):

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 41):

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            elif (chk == 5):

                l23 = abs(RESU_O2D_new(angles2c, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l23 = 0
        if (relation == 2):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_D2O_new(angles2c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 3):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 2):

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 41):

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            elif (chk == 5):

                l23 = abs(RESU_O2IP_new(angles2c, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l23 = 0
        if (relation == 4):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 3):

                l23 = abs(RESU_IP2O_new(angles2c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 41):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 5):

                l23 = abs(RESU_P2O_new(angles2c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l23 = 0
        if (relation == 5):
            ts2 = []
            ts3 = []
            ts22 = []
            ts33 = []
            chk = check_relation(angles2c, angles3c)
            if (chk != relation):
                l23 = 1
            if (chk == 1):

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 2):

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 3):

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            elif (chk == 4):

                l23 = abs(RESU_O2P_new(angles2c, angles3c))
            else:
                # print('Part of relation satisfied')
                l23 = 0


def loss13(relation):
    global l13
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    temp_relation = rel23
    run23(temp_relation)  # Problem, correct relation
    if (wecheck2[2] == 58):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 2):

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 3):

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 4):

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 41):

                l13 = abs(RESU_O2D_new(angles1, angles3))
            elif (chk == 5):

                l13 = abs(RESU_O2D_new(angles1, angles3))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_D2O_new(angles1, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 2):

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 4):

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 41):

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            elif (chk == 5):

                l13 = abs(RESU_O2IP_new(angles1, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 3):

                l13 = abs(RESU_IP2O_new(angles1, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 5):

                l13 = abs(RESU_P2O_new(angles1, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 2):

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 3):

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 4):

                l13 = abs(RESU_O2P_new(angles1, angles3))
            elif (chk == 41):

                l13 = abs(RESU_O2P_new(angles1, angles3))
            else:
                # print('Part of relation satisfied')
                l13 = 0

    elif (wecheck2[2] == 59):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 2):

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 3):

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 4):

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 41):

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            elif (chk == 5):

                l13 = abs(RESU_O2D_new(angles1c, angles3))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_D2O_new(angles1c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 2):

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 4):

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 41):

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            elif (chk == 5):

                l13 = abs(RESU_O2IP_new(angles1c, angles3))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 3):

                l13 = abs(RESU_IP2O_new(angles1c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 5):
                l13 = abs(RESU_P2O_new(angles1c, angles3))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 2):

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 3):

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 4):

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            elif (chk == 41):

                l13 = abs(RESU_O2P_new(angles1c, angles3))
            else:
                # print('Part of relation satisfied')
                l13 = 0

    if (wecheck2[2] == 60):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 2):

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 3):

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 4):

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 41):

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            elif (chk == 5):

                l13 = abs(RESU_O2D_new(angles1, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_D2O_new(angles1, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 2):

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 4):

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 41):

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            elif (chk == 5):

                l13 = abs(RESU_O2IP_new(angles1, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 3):

                l13 = abs(RESU_IP2O_new(angles1, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 5):

                l13 = abs(RESU_P2O_new(angles1, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 2):

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 3):

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 4):

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            elif (chk == 41):

                l13 = abs(RESU_O2P_new(angles1, angles3c))
            else:
                # print('Part of relation satisfied')
                l13 = 0

    if (wecheck2[2] == 61):
        if (relation == 1):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 2):

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 3):

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 4):

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 41):

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            elif (chk == 5):

                l13 = abs(RESU_O2D_new(angles1c, angles3c))
            else:
                # print('Disconnect relation satisfied')
                l13 = 0
        if (relation == 2):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_D2O_new(angles1c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 3):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 2):

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 4):

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 41):

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            elif (chk == 5):

                l13 = abs(RESU_O2IP_new(angles1c, angles3c))
            else:
                # print('Inverse Part of relation satisfied')
                l13 = 0
        if (relation == 4):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 3):
                abs(RESU_IP2O_new(angles1c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 41):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 5):

                l13 = abs(RESU_P2O_new(angles1c, angles3c))
            else:
                # print('Partial Overlap relation satisfied')
                l13 = 0
        if (relation == 5):
            ts1 = []
            ts3 = []
            ts11 = []
            ts33 = []
            chk = check_relation(angles1c, angles3c)
            if (chk != relation):
                l13 = 1
            if (chk == 1):

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 2):

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 3):

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 4):

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            elif (chk == 41):

                l13 = abs(RESU_O2P_new(angles1c, angles3c))
            else:
                # print('Part of relation satisfied')
                l13 = 0


def run12_1fix(relation):
    global l12
    global angles1, angles2, angles3, rel12, rel23, rel13
    global angles1c, angles2c, angles3c, wecheck2
    if (wecheck2[0] == 50):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect2(angles1, angles2)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1, angles2)
                ts11, ts22 = Overlap2Disconnect2(angles1, angles2)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect2(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect2(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1, angles2)
                ts11, ts22 = Overlap2Disconnect2(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1, angles2))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1, angles2)
                ts11, ts22 = Overlap2InversePartof2(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof2(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof2(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof2(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1, angles2)
                ts11, ts22 = Overlap2InversePartof2(angles1, angles2)
                with torch.no_grad():
                    angles1[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1, angles2)
                ts11, ts22 = Overlap2Partof2(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof2(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1, angles2)
                ts11, ts22 = Overlap2Partof2(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof2(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof2(angles1, angles2)
                with torch.no_grad():
                    angles2[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0

    elif (wecheck2[0] == 51):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect2(angles1c, angles2)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1c, angles2)
                ts11, ts22 = Overlap2Disconnect2(angles1c, angles2)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect2(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect2(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1c, angles2)
                ts11, ts22 = Overlap2Disconnect2(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1c, angles2))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1c, angles2)
                ts11, ts22 = Overlap2InversePartof2(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof2(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof2(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof2(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1c, angles2)
                ts11, ts22 = Overlap2InversePartof2(angles1c, angles2)
                with torch.no_grad():
                    angles1c[2] = angles2[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1c, angles2)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1c, angles2))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1c, angles2)
                ts11, ts22 = Overlap2Partof2(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof2(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1c, angles2)
                ts11, ts22 = Overlap2Partof2(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof2(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof2(angles1c, angles2)
                with torch.no_grad():
                    angles2[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0
        q1c = create_complement(angles1c)
        angles1 = torch.from_numpy(q1c)
        angles1.requires_grad = True

    elif (wecheck2[0] == 52):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect2(angles1, angles2c)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1, angles2c)
                ts11, ts22 = Overlap2Disconnect2(angles1, angles2c)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect2(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect2(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1, angles2c)
                ts11, ts22 = Overlap2Disconnect2(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1, angles2c))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1, angles2c)
                ts11, ts22 = Overlap2InversePartof2(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof2(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof2(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof2(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1, angles2c)
                ts11, ts22 = Overlap2InversePartof2(angles1, angles2c)
                with torch.no_grad():
                    angles1[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1, angles2c))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1, angles2c)
                ts11, ts22 = Overlap2Partof2(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof2(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1, angles2c)
                ts11, ts22 = Overlap2Partof2(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof2(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof2(angles1, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1, angles2c))
            else:
                # print('Part of relation satisfied')
                l12 = 0
        q1c = create_complement(angles2c)
        angles2 = torch.from_numpy(q1c)
        angles2.requires_grad = True

    elif (wecheck2[0] == 53):
        if (relation == 1):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)

            if (chk != relation):
                l12 = 1
            if (chk == 2):
                ts1, ts2 = Overlap2Disconnect2(angles1c, angles2c)

                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1c, angles2c)
                ts11, ts22 = Overlap2Disconnect2(angles1c, angles2c)

                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Disconnect2(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Disconnect2(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1c, angles2c)
                ts11, ts22 = Overlap2Disconnect2(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2D_new(angles1c, angles2c))
            else:
                # print('Disconnect relation satisfied')
                l12 = 0
        if (relation == 2):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_D2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 3):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1c, angles2c)
                ts11, ts22 = Overlap2InversePartof2(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2InversePartof2(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2InversePartof2(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2InversePartof2(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            elif (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1c, angles2c)
                ts11, ts22 = Overlap2InversePartof2(angles1c, angles2c)
                with torch.no_grad():
                    angles1c[2] = angles2c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2IP_new(angles1c, angles2c))
            else:
                # print('Inverse Part of relation satisfied')
                l12 = 0
        if (relation == 4):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_IP2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 41):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 5):

                ts1, ts2 = Partof2Overlap2(angles1c, angles2c)
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_P2O_new(angles1c, angles2c))
            else:
                # print('Partial Overlap relation satisfied')
                l12 = 0
        if (relation == 5):
            ts1 = []
            ts2 = []
            ts11 = []
            ts22 = []
            chk = check_relation(angles1c, angles2c)
            if (chk != relation):
                l12 = 1
            if (chk == 1):

                ts1, ts2 = Disconnect2Overlap2(angles1c, angles2c)
                ts11, ts22 = Overlap2Partof2(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 2):

                ts1, ts2 = Overlap2Partof2(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 3):

                ts1, ts2 = InversePartof2Overlap2(angles1c, angles2c)
                ts11, ts22 = Overlap2Partof2(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)
                C1.append(ts11)
                C2.append(ts22)
                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 4):

                ts1, ts2 = Overlap2Partof2(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2c))
            elif (chk == 41):

                ts1, ts2 = Overlap2Partof2(angles1c, angles2c)
                with torch.no_grad():
                    angles2c[2] = angles1c[2]/divide_by
                C1.append(ts1)
                C2.append(ts2)

                l12 = abs(RESU_O2P_new(angles1c, angles2))
            else:
                # print('Part of relation satisfied')
                l12 = 0
        q1c = create_complement(angles1c)
        angles1 = torch.from_numpy(q1c)
        angles1.requires_grad = True
        q1c = create_complement(angles2c)
        angles2 = torch.from_numpy(q1c)
        angles2.requires_grad = True


def check_special(angles1, angles2, status):
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
    if (status == 3):
        if (x1 == x2 and y1 == y2 and z1 == z2):
            if (r1 <= r2):
                chk = 3
                # print('3: First Chrome Part of Second Chrome, all X are Y')
    elif (status == 5):
        if (x1 == x2 and y1 == y2 and z1 == z2):
            if (r2 <= r1):
                chk = 5
                # print('4: Second Chrome Part of First Chrome')
    return chk


@app.route("/animate2")
def animate2():
    global main_count, q1, q2, q3, angles1, angles2, angles3, rel12, rel23, rel13, maxIter, maxRounds
    global lr1, lr2, l12, l23, l13, mastercount, isneg
    global angles1c, angles2c, angles3c, wecheck2
    global res
    mastercount = 0
    print("res is ", res)

    if (range(len(res) != 3)):
        return render_template('help.html', title='PythonIsHere!')

    list1 = [res[0]]
    list2 = [res[1]]
    list3 = [res[2]]

    global true_loss
    for itr1 in list1:
        for itr2 in list2:
            for itr3 in list3:
                main_count = main_count+1
                # if(main_count<42):
                #    continue
                print("=================", main_count,
                      "=====================\n")
                print("The values are: ", itr1, itr2, itr3)

                q1, q2, q3 = create_chromes()
                angles1 = torch.from_numpy(q1)
                angles1.requires_grad = True
                angles2 = torch.from_numpy(q2)
                angles2.requires_grad = True
                angles3 = torch.from_numpy(q3)
                angles3.requires_grad = True
                print("Chrome1 : ", q1)
                print("Chrome2 : ", q2)
                print("Chrome3 : ", q3)

                if (wecheck2[0] == 50):
                    pass
                elif (wecheck2[0] == 51):
                    q1c = create_complement(angles1)
                    angles1c = torch.from_numpy(q1c)
                    angles1c.requires_grad = True
                elif (wecheck2[0] == 52):
                    q1c = create_complement(angles2)
                    angles2c = torch.from_numpy(q1c)
                    angles2c.requires_grad = True
                elif (wecheck2[0] == 53):
                    q1c = create_complement(angles1)
                    angles1c = torch.from_numpy(q1c)
                    angles1c.requires_grad = True
                    q1c = create_complement(angles2)
                    angles2c = torch.from_numpy(q1c)
                    angles2c.requires_grad = True

                if (wecheck2[1] == 54):
                    pass
                elif (wecheck2[1] == 55):
                    q1c = create_complement(angles2)
                    angles2c = torch.from_numpy(q1c)
                    angles2c.requires_grad = True
                elif (wecheck2[1] == 56):
                    q1c = create_complement(angles3)
                    angles3c = torch.from_numpy(q1c)
                    angles3c.requires_grad = True
                elif (wecheck2[1] == 57):
                    q1c = create_complement(angles2)
                    angles2c = torch.from_numpy(q1c)
                    angles2c.requires_grad = True
                    q1c = create_complement(angles3)
                    angles3c = torch.from_numpy(q1c)
                    angles3c.requires_grad = True

                if (wecheck2[2] == 58):
                    pass
                elif (wecheck2[2] == 59):
                    q1c = create_complement(angles1)
                    angles1c = torch.from_numpy(q1c)
                    angles1c.requires_grad = True
                elif (wecheck2[2] == 60):
                    q1c = create_complement(angles3)
                    angles3c = torch.from_numpy(q1c)
                    angles3c.requires_grad = True
                elif (wecheck2[2] == 61):
                    q1c = create_complement(angles1)
                    angles1c = torch.from_numpy(q1c)
                    angles1c.requires_grad = True
                    q1c = create_complement(angles3)
                    angles3c = torch.from_numpy(q1c)
                    angles3c.requires_grad = True

                table = np.zeros((4, 6))
                table[0][2] = 1
                table[2][4] = 1
                table[0][4] = 1

                for i in range(4):
                    for j in range(6):
                        if (table[i][j] != 0):
                            if (i == 0 and j == 2):
                                rel12 = table[i][j]
                            if (i == 1 and j == 2):
                                rel12 = table[i][j]
                            if (i == 0 and j == 3):
                                rel12 = table[i][j]
                            if (i == 1 and j == 3):
                                rel12 = table[i][j]
                            if (i == 2 and j == 4):
                                rel23 = table[i][j]
                            if (i == 3 and j == 4):
                                rel23 = table[i][j]
                            if (i == 2 and j == 5):
                                rel23 = table[i][j]
                            if (i == 3 and j == 5):
                                rel23 = table[i][j]
                            if (i == 0 and j == 4):
                                rel13 = table[i][j]
                            if (i == 1 and j == 4):
                                rel13 = table[i][j]
                            if (i == 0 and j == 5):
                                rel13 = table[i][j]
                            if (i == 1 and j == 5):
                                rel13 = table[i][j]

                gLoss = 1
                rounds = 0

                lr1 = 0.001
                lr2 = 0.01
                # Change r1 and r2 for angles1, angles2 and angles3
                l12 = 1
                l23 = 1
                l13 = 1
                while (rounds < maxRounds and gLoss > 0):
                    relation = rel12
                    run12(relation)
                # ===============================================================================
                    relation = rel23
                    # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                    run23(relation)
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
                        if (table[i][j] != 0):
                            if (i == 0 and j == 2):
                                rel12 = table[i][j]
                            if (i == 1 and j == 2):
                                rel12 = table[i][j]
                            if (i == 0 and j == 3):
                                rel12 = table[i][j]
                            if (i == 1 and j == 3):
                                rel12 = table[i][j]
                            if (i == 2 and j == 4):
                                rel23 = table[i][j]
                            if (i == 3 and j == 4):
                                rel23 = table[i][j]
                            if (i == 2 and j == 5):
                                rel23 = table[i][j]
                            if (i == 3 and j == 5):
                                rel23 = table[i][j]
                            if (i == 0 and j == 4):
                                rel13 = table[i][j]
                            if (i == 1 and j == 4):
                                rel13 = table[i][j]
                            if (i == 0 and j == 5):
                                rel13 = table[i][j]
                            if (i == 1 and j == 5):
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
                while (rounds < maxRounds and gLoss > 0):
                    relation = rel12
                    run12(relation)
                # ===============================================================================
                    relation = rel23
                    # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                    run23(relation)
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
                while (rounds < maxRounds and gLoss > 0):
                    relation = rel12
                    run12(relation)
                # ===============================================================================
                    relation = rel23
                    # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                    run23(relation)
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
                while (rounds < maxRounds and gLoss > 0):
                    relation = rel12
                    run12_1fix(relation)
                # ===============================================================================
                    relation = rel23
                    # wrong funtion, it shiuld be 2 and 3 in order and never 3 and 2
                    run23(relation)
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
                print(check_relation(angles1, angles2), check_relation(
                    angles2, angles3), check_relation(angles1, angles3))
                print("Actual loss for given orientation is :", true_loss)
                print("\n======================================")
                q11, q21, q31 = create_chromes()
                angles11 = torch.from_numpy(q11)
                angles11.requires_grad = True
                if (true_loss != 0 and itr1 == check_special(angles11, angles11, itr1) and itr2 == check_special(angles11, angles11, itr2) and itr3 == check_special(angles11, angles11, itr3)):
                    true_loss = 129
                    print('special case detected and loss is 0')
                else:
                    pass
                print("\n======================================")
    save_1 = []
    save_2 = []
    save_3 = []
    save_1c = []
    save_2c = []
    save_3c = []
    save_4 = []
    save_5 = []
    q1c = create_complement(angles1)
    angles1c = torch.from_numpy(q1c)
    angles1c.requires_grad = True
    q1c = create_complement(angles2)
    angles2c = torch.from_numpy(q1c)
    angles2c.requires_grad = True
    q1c = create_complement(angles3)
    angles3c = torch.from_numpy(q1c)
    angles3c.requires_grad = True
    with torch.no_grad():
        tp1 = angles1
        tp2 = angles2
        tp3 = angles3
        tp1c = angles1c
        tp2c = angles2c
        tp3c = angles3c
        np_arr1 = tp1.numpy()
        np_arr2 = tp2.numpy()
        np_arr3 = tp3.numpy()
        np_arr1c = tp1c.numpy()
        np_arr2c = tp2c.numpy()
        np_arr3c = tp3c.numpy()
        np_loss = np.array([true_loss])
        print(true_loss)
        ls1 = []
        ls2 = []
        ls3 = []
        ls4 = []
        ls1 = np_arr1.tolist()
        ls2 = np_arr2.tolist()
        ls3 = np_arr3.tolist()
        ls1c = np_arr1c.tolist()
        ls2c = np_arr2c.tolist()
        ls3c = np_arr3c.tolist()
        ls4 = np_loss.tolist()
        save_1.append(ls1)
        save_2.append(ls2)
        save_3.append(ls3)
        save_1c.append(ls1c)
        save_2c.append(ls2c)
        save_3c.append(ls3c)
        ls4.append(0)
        ls4.append(0)
        save_4.append(ls4)
        ls5 = [mastercount, mastercount, mastercount]
        save_5.append(ls5)
    df1 = pd.DataFrame(save_1, columns=['theta1', 'phi1', 'alpha1'])
    df2 = pd.DataFrame(save_2, columns=['theta2', 'phi2', 'alpha2'])
    df3 = pd.DataFrame(save_3, columns=['theta3', 'phi3', 'alpha3'])
    df1c = pd.DataFrame(save_1c, columns=['theta1c', 'phi1c', 'alpha1c'])
    df2c = pd.DataFrame(save_2c, columns=['theta2c', 'phi2c', 'alpha2c'])
    df3c = pd.DataFrame(save_3c, columns=['theta3c', 'phi3c', 'alpha3c'])
    df4 = pd.DataFrame(save_4, columns=['loss', 'temp1', 'temp2'])
    df5 = pd.DataFrame(save_5, columns=['iter1', 'iter2', 'iter3'])
    print(save_5)
    global strval, stepval
    strvals = []
    stepvals = []
    strvals.append(strval)
    dfstr = pd.DataFrame(strvals, columns=['s1', 's2', 's3'])
    stepvals = []
    stepvals.append(stepval)
    dfstep = pd.DataFrame(stepvals, columns=['step1', 'step2', 'step3'])
    nlss = []
    ls = [1, 2, 3]
    nls = [1, 2, 3]
    for i in ls:
        if (i not in isneg):
            nls[i-1] = 0
    nlss.append(nls)
    dfnls = pd.DataFrame(nlss, columns=['notS', 'notM', 'notP'])
    df_concat = pd.concat(
        [df1, df2, df3, df1c, df2c, df3c, df4, df5, dfstr, dfstep, dfnls], axis=1)
    print(df_concat)

    # new_df.columns =['X_axis', 'Y_axis', 'Z_axis', 'Radius']
    json_df_concat = df_concat.to_json(orient='records')
    # json_df2 = df2.to_dict(orient='records')
    print('Iterations:', mastercount)

    return render_template('withcomp.html', title='PythonIsHere!', JSON_data=json_df_concat)




# ====================
