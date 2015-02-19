import subprocess
averageTime = 0

maxMatrix = 20
averageThread = 20
matrixInterval = 4

thefile = open('myfile.dat', 'w+')


for matrixSize in range(4,maxMatrix + 1,1):
  averageTime = 0
  for average in range(0, averageThread):
    value = str(matrixSize)
    cmd = ["./new_test", value, value, value, value]
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    thefile.write("Size: %s\n" % value);
    thefile.write("%s\n\n" % output);
    print matrixSize
