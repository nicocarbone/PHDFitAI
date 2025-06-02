import numpy as np
import MC_PHD as mcphd
from matplotlib import pyplot as plt
import time
import csv
import sys

nPoints = 10

if len(sys.argv) > 1:
    try:
        nPoints = int(sys.argv[1])
    except ValueError:
        print("Invalid argument for nPoints. Using default value:", nPoints)

upsRange = [0.1, 2.0]
uaRange = [0, 0.02]

g = 0.9
usRange = np.asarray(upsRange)/(1-g) 

n = 1.4
sdSep = 30

nPhotons = 5e8
nBins = 4096
maxTime = 25e-9

fileName = "SIMs/rho{}mm/mcSims_V1.dat".format(int(sdSep))

for i in range(nPoints):
    uaBulk = np.random.uniform(uaRange[0], uaRange[1])
    usBulk = np.random.uniform(usRange[0], usRange[1])
    
    print("\n_______________________________________________________________________")
    print(f"Simulation {i+1}/{nPoints}: uaBulk = {uaBulk:.4f}, usBulk = {usBulk:.4f}\n")
    
    
    # Run the MCPHD simulation
    tic = time.time()
    histo, timeParams = mcphd.MCPHD(uaBulk=uaBulk, usBulk=usBulk, g=g, n=n, sdSep=sdSep, slabThickness=100, detRad=3, isRefl=True, maxTime=maxTime, nPhotons=nPhotons, nTimeBins=nBins)
    toc = time.time()
    
    print(f"Simulation {i+1}/{nPoints} completed in {toc - tic:.2f} seconds")
    
    # Combine ua, ups, and the flattened results into a single list
    row_data = [uaBulk, usBulk] + histo[0].flatten().tolist()

    with open(fileName, 'a', newline='') as f: # 'a' for append mode, newline='' for consistent line endings
        writer = csv.writer(f)
        writer.writerow(row_data)


