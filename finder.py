from PIL import Image
import PIL
import numpy as np
from random import *
from random import choice
from math import ceil
import time


def resize(img,height,width):
    img = img.resize((width,height), PIL.Image.ANTIALIAS)
    return img

def resize_noRatio(img,baseheight):
    img = img.resize((baseheight, baseheight), PIL.Image.ANTIALIAS)
    return img

def Randomize(bib_files,background):
    num_bibs = randrange(1,4)
    chosen_grid=[]
    bibs = []
    ## Select a random number of bibs to put on to a 19x19 grid
    for x in range(num_bibs):
        bib_file = randrange(0,len(bib_files))
        area_x,area_y = random(),random()
        choose = choice([i for i in range(0,48) if i not in chosen_grid])
        chosen_grid.append(choose)
        #Randomly determine bib height and width
        img_width = randrange(50,75)/448
        img_height = img_width*(uniform(0.5,0.8))
        bibs.append((choose,bib_files[bib_file],area_x,area_y,img_width,img_height))
    
    ##Choose a random background image, load it (resize it before loading)
    back_num = randrange(0,len(background))
    back_image = Image.open(background[back_num])
    back_image = resize(back_image,448,448)
    
    ## Paste the selected bibs on to the background image using the grid values
    for bib in bibs:
        ## randomly resize the bibs
        bib_img = Image.open(bib[1])
        width = bib[4]*448
        height = bib[5]*448
        bib_img=resize(bib_img,int(width),int(height))
        
        grid_y = ceil(bib[0]/7)
        grid_x = bib[0]-((grid_y-1)*7)
        #Actual Area for the bib to be pasted on 
        area_x = (grid_x-1)*64 - bib[2]*64*0.5
        area_y = (grid_y-1)*64 - bib[3]*64*0.5
        
        back_image.paste(bib_img,(int(area_x),int(area_y)))
    
    ## Create the output matrix for classification
    cells = [i[0] for i in bibs]
    array=[]
    for y in range(0,7):
        row = []
        for x in range(1,8):
            cell_num = y*7 + x
            if cell_num in cells:
                bib = [x for x in bibs if x[0]==cell_num]
                point = np.array([1,bib[0][4],bib[0][5],bib[0][2],bib[0][3]])
            else:
                point = np.zeros((5))
            row.append(point)
        array.append(row)
        
    array = np.array(array)
    
    return array, back_image
        


bib_files = ['bib.png','bib1.png','bib2.png','bib3.png','bib4.png','bib5.png','bib6.png','bib7.png']
background = ['back1.jpg','back2.jpg','back3.jpg','back4.jpg','back5.jpg','back6.jpg','back7.jpg']

training_set = 32
batches = 5
id=0


tick =time.time()
for batch in range(batches):
    i=0
    images = []
    output = []
    while i < training_set:
    
        array,back_image = Randomize(bib_files,background)
        images.append(np.array(back_image))
        output.append(array)
        i+=1
        
        if i % 100 == 0:
            print(time.time()-tick,'s for batch',i)
            tick = time.time()

    output = np.array(output)
    images = np.array(images)
    
    np.save('output/output'+str(id),output)
    np.save('images/images'+str(id),images)
    
    id+=1

#back_image.save('test.jpg')
#output = np.array(output)
#images = np.array(images)
#
#np.save('output',output)
#np.save('images',images)