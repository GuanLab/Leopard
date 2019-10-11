
def scaleImage (image,scale):
    import cv2
    [x,y]= image.shape
    x1=x
    y1=int(round(y*scale))
    image=cv2.resize(image.astype('float32'),(y1,x1)) # check this for multiple channnels!!
    new=np.zeros((x,y))
    if (y1>y):
        start=int(round(y1/2-y/2))
        end=start+y
        new=image[:,start:end]
    else:
        new_start=int(round(y-y1)/2)
        new_end=new_start+y1
        new[:,new_start:new_end]=image
    return new


def pool_avg_2(input,if_mask=False):
    index1=np.arange(0,input.shape[1],2)
    index2=np.arange(1,input.shape[1],2)
    if (len(index2)<len(index1)):
        index2=np.concatenate((index2,[input.shape[1]-1]))
    output = (input[:,index1] + input[:,index2]) / float(2)
    if (if_mask): # -1 position are masked by -1, not avg
        mask = np.minimum(input[:,index1],input[:,index2])
        output[mask<0]=-1
    return output

