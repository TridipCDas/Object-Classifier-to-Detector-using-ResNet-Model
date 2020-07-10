import imutils

def sliding_window(img,steps,ws):
    '''
    Returns windows sliding over the entire image
    
    Arguments:
    img:input image
    steps: No. of pixels we want to skip in (x,y) directions
    ws: Size(width,height) of  extracted windows from the image
    
    '''
    
    
    #Rolling over the Rows
    for y in range(0,img.shape[0]-ws[1],steps):
        #Rolling over the columns
        for x in range(0,img.shape[1]-ws[0],steps):
            
            #yield the current window
            yield(x,y,img[y:y+ws[1],x:x+ws[0]])


def image_pyramid(img,scale=1.5, minSize=(224, 224)):
    '''
    Returns the image pyramid of the image
    
    Arguments:
    img: Input image
    scale=Scaling factor of the image at each step
    minSize: Minimum size of the output image(image layer of the pyramid)
    
    '''
    #At first,return the original image
    yield img
    
    while True:
        
        #Dimensions of the next image
        width=int(img.shape[1]/scale)
        image=imutils.resize(img,width=width)
        
        #Checking the minimum window conditions
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        #Yield the next image    
        yield image
    
        
