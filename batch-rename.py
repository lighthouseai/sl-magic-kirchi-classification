import os
from datetime import date
import time
import random

dirName = "./empty/"

newName = str(date.today())+"-"+str(time.time())

for i in os.listdir(dirName):
    os.rename(dirName+i,dirName+newName+"-"+str(random.random()*100)+".jpg")