import numpy as np

def uxy_to_rot(ux,uy):
    dux_dx, dux_dy = np.gradient(ux, edge_order=1)
    duy_dx, duy_dy = np.gradient(uy, edge_order=1)
    vorticity = dux_dy - duy_dx
    return vorticity

def vort_to_color(vorticity, wall):
    #rot = vorticity / np.max(np.abs(vorticity)) 
    color_rot = single_channel_to_three_channel(vorticity)
    color_rot = np.log(1+color_rot)
    #color_rot /= 7
    color_rot[wall>0,:] = [0.5, 0.7, 0.5]
    return color_rot

def uxy_to_color(uxts,uyts,wallts):
    vorticity = uxy_to_rot(uxts.detach().cpu().numpy(),uyts.detach().cpu().numpy())
    return vort_to_color(vorticity,wallts.detach().cpu().numpy())

def single_channel_to_three_channel(data):
    result = np.zeros((data.shape[0], data.shape[1],3))
    positive = data.copy()
    positive[positive<0] = 0
    negative = data.copy()
    negative[negative>0] = 0
    result[:,:,0] += positive
    result[:,:,2] += -negative
    return result