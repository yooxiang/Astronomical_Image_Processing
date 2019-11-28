#%%
#function for arranging pixels chosen in descending order
def BrightnessDsc(dataset):
    #split data set in blocks of 29x10
    import numpy as np
    from scipy import stats
    brightness_all=[]
    MeanMatrix=np.zeros([29,10])
    SMatrix=np.zeros([29,10])
    for r in range(2):#iterates for each CELL row (number of rows os 29)
        for c in range(2):#iterates through required CELL column(number of columns is 10)
            brightness=[]
            brightness_row=[]
            brightness_column=[]
            for i in range ((c*257),((c+1)*257)):
                for j in range ((r*159),(r+1)*159):
  
                    brightness.append(dataset[j][i])
                    brightness_row.append(j)
                    brightness_column.append(i)
                    #plt.hist(dataseti,normed=True,bins=50)
            m, s = stats.norm.fit(brightness) # get mean and standard deviation  
            MeanMatrix[r][c]=m
            SMatrix[r][c]=s
            brightness_chosen=[]
            brightness_row_chosen=[]
            brightness_column_chosen =[]
            for i in range(len(brightness)):
                    if brightness[i]>m+2*s:
                        positionr=brightness_row[i]
                        positionc= brightness_column[i]
                        brightness_chosen.append(brightness[i])
                        brightness_row_chosen.append(positionr)
                        brightness_column_chosen.append(positionc)
    
#            print(len(brightness_chosen),brightness_chosen[-1])
            for i in range(len(brightness_chosen)):
                #find max value of brightness indicating position of galaxy and record its count and position
                brightness_x_y=[]#[pixel count,rowposition,columnposition]
                maxcount=max(brightness_chosen)
                inipos=brightness_chosen.index(maxcount)
                brightness_x_y.append(brightness_chosen[inipos])
                brightness_x_y.append(brightness_row_chosen[inipos])
                brightness_x_y.append(brightness_column_chosen[inipos])
                del brightness_chosen[inipos]
                del brightness_row_chosen[inipos]
                del brightness_column_chosen[inipos]
                brightness_all.append(brightness_x_y)
    brightness_all.sort(reverse=True)
    return (brightness_all,MeanMatrix,SMatrix,dataset)

#%%
#function for getting points in circular pattern around galaxy
import scipy as sp
import matplotlib.pyplot as plt

def f_circle(x, y, x_0,y_0,r): #circle function
    return (x-x_0)**2 + (y-y_0)**2 - r**2

               
def d(x, y, x_0, y_0, r): #d funtion from website
    return f_circle((x-0.5),(y+1),x_0,y_0,r)

def check(x,y,x_store,y_store,x_y_store):

    
    if [x,y] in x_y_store:
            return False
    else:
            return True
            x_y_store.append([x,y])
        


def circle(x_0, y_0,r): #function which returns list of x and list f y
    
    x_shift = x_0 
    y_shift = y_0
    
    x_0 = 0
    y_0 = 0
    
    x_store = [(x_0)+r]
    y_store = [y_0]
    x_y_store = ([[(x_0)+r,y_0]])
    
    if check(y_0,(x_0+r),x_store,y_store,x_y_store) == True:
        x_store.append(y_0)
        y_store.append((x_0)+r)
        x_y_store.append([y_0,(x_0)+r])
           
    if check(y_0,-((x_0)+r),x_store,y_store,x_y_store) == True:    
        x_store.append(y_0)
        y_store.append(-((x_0)+r))
        x_y_store.append([y_0,-((x_0)+r)])

    if check((x_0)+r,-y_0,x_store,y_store,x_y_store) == True:        
        x_store.append((x_0)+r)
        y_store.append(-y_0)
        x_y_store.append([((x_0)+r),-y_0])
     
    if check(-((x_0)+r),-y_0,x_store,y_store,x_y_store) == True:       
        x_store.append(-((x_0)+r))
        y_store.append(-y_0)
        x_y_store.append([-((x_0)+r),-y_0])
    
 
    if check(-y_0,-((x_0)+r),x_store,y_store,x_y_store) == True:        
        x_store.append(-y_0)
        y_store.append(-((x_0)+r))
        x_y_store.append([-y_0,-((x_0)+r)])


    if check(-y_0,(x_0 + r),x_store,y_store,x_y_store) == True:         
        x_store.append(-y_0)
  

        y_store.append((x_0)+r)
        x_y_store.append([-y_0,((x_0)+r)])
    

    if check(-((x_0)+r),y_0,x_store,y_store,x_y_store) == True:        
        x_store.append(-((x_0)+r))
        y_store.append(y_0)
        x_y_store.append([-((x_0)+r),y_0])
    
    
    x = x_0 + r
    y = y_0

    while x > y:
        if d(x,y,x_0,y_0,r) > 0:
            x = x - 1
            y = y + 1
            
            if check(x,y,x_store,y_store,x_y_store) == True:
                x_store.append(x)
                y_store.append(y)
                x_y_store.append([x,y])
            
            if check(y,x,x_store,y_store,x_y_store) == True:
                x_store.append(y)
                y_store.append(x)
                x_y_store.append([y,x])
            
            if check(y,-x,x_store,y_store,x_y_store) == True:
                x_store.append(y)
                y_store.append(-x)
                x_y_store.append([y,-x])
            
            if check(x,-y,x_store,y_store,x_y_store) == True:
                x_store.append(x)
                y_store.append(-y)
                x_y_store.append([x,-y])
            
            if check(-x,-y,x_store,y_store,x_y_store) == True:
                x_store.append(-x)
                y_store.append(-y)
                x_y_store.append([-x,-y])
                
            if check(-y,-x,x_store,y_store,x_y_store) == True:
                x_store.append(-y)
                y_store.append(-x)
                x_y_store.append([-y,-x])
            
            if check(-y,x,x_store,y_store,x_y_store) == True:
                x_store.append(-y)
                y_store.append(x)
                x_y_store.append([-y,x])
            
            if check(-x,y,x_store,y_store,x_y_store) == True:
                x_store.append(-x)
                y_store.append(y)
                x_y_store.append([-x,y])
            
        
        if d(x,y,x_0,y_0,r) < 0:
            y = y + 1
    
            if check(x,y,x_store,y_store,x_y_store) == True:
                x_store.append(x)
                y_store.append(y)
                x_y_store.append([x,y])
            
            if check(y,x,x_store,y_store,x_y_store) == True:
                x_store.append(y)
                y_store.append(x)
                x_y_store.append([y,x])
            
            if check(y,-x,x_store,y_store,x_y_store) == True:
                x_store.append(y)
                y_store.append(-x)
                x_y_store.append([y,-x])
            
            if check(x,-y,x_store,y_store,x_y_store) == True:
                x_store.append(x)
                y_store.append(-y)
                x_y_store.append([x,-y])
            
            if check(-x,-y,x_store,y_store,x_y_store) == True:
                x_store.append(-x)
                y_store.append(-y)
                x_y_store.append([-x,-y])
                
            if check(-y,-x,x_store,y_store,x_y_store) == True:
                x_store.append(-y)
                y_store.append(-x)
                x_y_store.append([-y,-x])
            
            if check(-y,x,x_store,y_store,x_y_store) == True:
                x_store.append(-y)
                y_store.append(x)
                x_y_store.append([-y,x])
            
            if check(-x,y,x_store,y_store,x_y_store) == True:
                x_store.append(-x)
                y_store.append(y)
                x_y_store.append([-x,y])
                
      
    for i in range(len(x_store)):
        x_store[i] = x_store[i] + x_shift
        y_store[i] = y_store[i] + y_shift
    return sp.array(x_store), sp.array(y_store)


def circle_plot(x_0,y_0,r): #function which returns plot of cirlce points
    c  = circle(x_0,y_0,r)
    plt.scatter(c[0],c[1])
    
#%%
#function that calculates calibrated magnitude
def CountToBright(counttotal,counttotalerr):
    import math
    from astropy.io import fits
    hdulist= fits.open("A1_mosaic.fits")

    inst0point=hdulist[0].header["MAGZPT"]#calibration magnitude
    inst0pointerr=hdulist[0].header["MAGZRR"]#calibration magnitude error
    
    mag=inst0point-2.5*math.log(counttotal,10)#conversion of counts to calibrated magnitude
    magierr=(-2.5*counttotalerr)/(math.log(10)*counttotal)#error of 2nd component in expression
    magerr=(inst0pointerr**2+magierr**2)**0.5#error of magnitude
    return (mag,magerr)
#%%
'''
returns a list
first entry corrected cumulative brightness
second radius
row
column
'''
def galaxy_finder(data,Range): # put brightness(data) in 

    average_plot_first = []
    average_plot_last = []
    radius_plot_first = []
    
    brightness_all = data[0] # has brightness, x, y in desecnding order


    Matrix = data[3]

    
    brightness_total_c = brightness_all[0][0]#add the middle pixel THIS IS JUST BRIGHTNESS
    pixel_count_c = 1
    
    centre = brightness_all[0][2],brightness_all[0][1]
    #entre = 40,50
    #print("centre", centre)
    
    brightness_loop = 0
    pixel_loop = 0
    
    average = []
    
    x = []
    y = []
    
    for i in range(1, Range):
        brightness_loop = 0
       
        pixel_loop = len(circle(centre[0],centre[1],i)[0])

        for j in range(len(circle(centre[0],centre[1],i)[0])): # counts how mnay points in each circle
            brightness_loop += Matrix[circle(centre[0],centre[1],i)[1][j]][circle(centre[0],centre[1],i)[0][j]] # take column first!!
            brightbefore=Matrix[circle(centre[0],centre[1],i)[1][j]][circle(centre[0],centre[1],i)[0][j]]
            Matrix[circle(centre[0],centre[1],i)[1][j]][circle(centre[0],centre[1],i)[0][j]]=0 #sets brightness to zerp
            if [brightbefore,circle(centre[0],centre[1],i)[1][j],circle(centre[0],centre[1],i)[0][j]] in brightness_all:
                position=brightness_all.index([brightbefore,circle(centre[0],centre[1],i)[1][j],circle(centre[0],centre[1],i)[0][j]]) #finds position from data
                del brightness_all[position] #deletes brigthnees x and y
        
        
        pixel_count_c = pixel_count_c + pixel_loop
        
       
        brightness_total_c = brightness_total_c + brightness_loop

        average.append(brightness_loop/pixel_loop)
        
        y.append(brightness_total_c)
        x.append(pixel_count_c)
    
    Matrix[brightness_all[0][1]][brightness_all[0][2]]=0
    position=brightness_all.index([brightness_all[0][0],brightness_all[0][1],brightness_all[0][2]])        
    del brightness_all[position]
        
       
    
    for i in range (4):
        average_plot_first.append(average[i])
        radius_plot_first.append(i+1)
    
    for i in range(Range,Range -5,-1):

        average_plot_last.append(average[i-2])

    
    popt = sp.polyfit(radius_plot_first,average_plot_first,1)

    Fitted1 = sp.poly1d(popt)

    '''
    delete later below
    '''
    
    
    plt.figure()
    plt.scatter(x,sp.log(y),marker  = 'x')
    plt.xlabel("Cumulative Pixel Count")
    plt.ylabel("Cumulaitve Brightness")
    plt.title("Cumulative Brightness")
    plt.savefig("cumuati", dpi = 1000)
    
    plt.figure()
    plt.scatter(sp.linspace(1,len(x)+1,len(x)),average,marker = 'x')
    #|plt.plot(sp.linspace(0,len(x),len(x)), Fitted1(sp.linspace(0,len(x),len(x))))
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Average Brigthness")
    plt.title("Average Brightness of Pixels as Function of Radius for one Galaxy")
    plt.savefig("avergae_birght",dpi = 1000)
    plt.ylim((min(average)-50, max(average)+50))
    
    last_average = sp.mean(average_plot_last)

    average_subtracted = []
  
    for i in range(len(average)):

        average_subtracted.append(average[i] - last_average)

    #plt.figure()
    #plt.scatter(sp.linspace(1,len(x)+1,len(x)),average_subtracted)
    
    average = average_subtracted[0]
    radius = 0

  
    while average > 100:
        radius += 1
        
        average = average_subtracted[radius]
    corrected_cumulative = y[radius] - x[radius]*last_average
    

    return corrected_cumulative,radius,centre[0],centre[1]
#%%
def galaxy_looper(data,Range):
    galaxies = []
  
    while len(data[0]) != 0:
        x = galaxy_finder(data,Range)
        
        if x[1] == 0:
            print("false galaxy")
        else:
            galaxies.append(x)
            print("new galaxy",len(data[0]))
        
    return galaxies,len(galaxies)
#%%
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import stats
import scipy as sp
hdulist= fits.open("A1_mosaic.fits")
dataset=hdulist[0].data
galaxy_finder(BrightnessDsc(dataset),30)
