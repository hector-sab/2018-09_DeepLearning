import subprocess
import os

#se definen las varables
path="/home/dagopa/Desktop/"
filename="test.png"
job="20181020-084216-ec95"

#Se llama al servicio web
bashCommand = "curl localhost:5000/models/images/generic/infer_one.json -XPOST -F job_id=" + job + " -F snapshot_epoch=29 -F image_file=@" + path + filename

process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#Se guarda el resultado en un archivo json
print(filename)
with open(path + filename + '.json','w') as f:    
    f.write(output)

