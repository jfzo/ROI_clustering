step1 = lambda x: 'Configuracion nnodos' in  x
step2 = lambda x: 'Kmeans Experiments' in x
step3 = lambda x: 'SNN Experiments' in x

with open('dcm_snr.log') as f:
    linea = f.readline().strip()
    while len(linea) > 0:
        if step1(linea):
            print linea
        elif step2(linea):
            print "Kmeans",f.readline().strip()
        elif step3(linea):
            f.readline()
            print "SNN",f.readline().strip()
        linea = f.readline().strip()
