import subprocess
averageTime = 0

maxMatrix = 400
averageThread = 20
matrixInterval = 4
maxThread = 64

thefile = open('myfile.dat', 'w+')


for matrixSize in range(20,maxMatrix + 1,1):
  value = str(matrixSize)
  threadStr = str(maxThread);
  cmd = ["./new_test", value, value, value, value, threadStr]
  output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
  print output
