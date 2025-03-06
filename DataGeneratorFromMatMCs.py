import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import IRF_Simulator as irfsim
from scipy.interpolate import interp1d
import scipy.signal as sig
import time
import mat73
import time


tic = time.time()

mcData = mat73.loadmat('DTOFs_2-24-2025_fine.mat')
#mcData = mat73.loadmat('DTOFs_2-25-2025_coarseAndWider_1e9.mat')
uas = mcData['ua'].reshape(-1)
upss = mcData['ups'].reshape(-1)
rhos = mcData['rho'].reshape(-1)
dtofs = mcData['DTOF']
cfg = mcData['cfg']

nroOPs = 750
nroRhos = 1
nroIRFs = 2
n_channels = 4096

irf_timeRange_ns = 50
irf_timeResolution_ps = (irf_timeRange_ns/n_channels)*1000
irf_minDelay = 1000

irf_real_proportion = 0.7
irf_real_samples = np.load('irfs_samples.npy')

sim_results = []
sim_irfs = []
sim_tags = []
sim_randomsPHD = []
sim_randomsIRF = []

print('Starting simulation. Number of datapoints: {}...'.format(nroOPs*nroRhos*nroIRFs))

idsim = 0

for iua in range(nroOPs):
    for irho in range(nroRhos):
        
        idua = np.random.randint(0, len(uas))
        idups = np.random.randint(0, len(upss))
        idrho = np.random.randint(0, len(rhos))
        
        if nroRhos == 1:
            idrho = 1
        
        ua = uas[idua]
        ups = upss[idups]
        rho = rhos[idrho]
        
        dtof = dtofs[:, idua, idups, idrho]
        
        # Get the current size of dtof
        current_size = dtof.shape[0]

        # Create an array of indices for the current and new sizes
        current_indices = np.arange(current_size)
        new_indices = np.linspace(0, current_size - 1, n_channels)

        # Create an interpolation function
        interpolation_function = interp1d(current_indices, dtof, kind='cubic')

        # Interpolate dtof to the new size
        dtof_interpolated = interpolation_function(new_indices)
        
        for iirf in range(nroIRFs):
            
            idsim += 1
            
            print('{}/{} - ua = {}, ups = {}, rho = {}'.format(idsim, nroOPs*nroRhos*nroIRFs, ua, ups, rho))
            
            irf_real_selected = np.random.rand()
            if irf_real_selected < irf_real_proportion:
                irf = irf_real_samples[np.random.randint(0, len(irf_real_samples))]
            else:
                irf_peak_delay1 = int(np.random.normal(5000, 500))
                if irf_peak_delay1 < irf_minDelay: irf_peak_delay1 = irf_minDelay
                irf_peak_delay2 = irf_peak_delay1 + int(np.random.normal(200, 50))
                if irf_peak_delay2 < irf_peak_delay1: irf_peak_delay2 = irf_peak_delay1
                irf_peak_delay3 = irf_peak_delay2 + int(np.random.normal(600, 50))
                if irf_peak_delay3 < irf_peak_delay2: irf_peak_delay3 = irf_peak_delay2
                irf_peak_delay4 = irf_peak_delay3 + int(np.random.normal(200, 50))
                if irf_peak_delay4 < irf_peak_delay3: irf_peak_delay4 = irf_peak_delay3
                
                irf_peak_ratio1 = 1
                irf_peak_ratio2 = np.random.normal(0.6, 0.1)
                if irf_peak_ratio2 < 0: irf_peak_ratio2 = 0
                irf_peak_ratio3 = np.random.normal(0.3, 0.1)
                if irf_peak_ratio3 < 0: irf_peak_ratio3 = 0
                irf_peak_ratio4 = np.random.normal(1, 0.1)
                if irf_peak_ratio4 < 0: irf_peak_ratio4 = 0
                
                irf_peak_width_ps1 = int(np.random.normal(350, 50))
                if irf_peak_width_ps1 < 0: irf_peak_width_ps1 = 0
                                    
                irf_peak_width_ps2 = int(np.random.normal(550, 50))
                if irf_peak_width_ps2 < irf_peak_width_ps1: irf_peak_width_ps2 = irf_peak_width_ps1
                
                irf_peak_width_ps3 = int(np.random.normal(700, 100))
                if irf_peak_width_ps3 < irf_peak_width_ps2: irf_peak_width_ps3 = irf_peak_width_ps2
                
                irf_peak_width_ps4 = int(np.random.normal(2500, 200))
                if irf_peak_width_ps4 < irf_peak_width_ps3: irf_peak_width_ps4 = irf_peak_width_ps3
                
                irf_jitter_std_dev_ps = np.random.normal(30, 5)
                
                irf_detector_response_fwhm_ps = np.random.normal(40, 5)
                
                irf_avg_noise_floor = int(np.random.normal(200, 20))
                if irf_avg_noise_floor < 0: irf_avg_noise_floor = 0
                
                irf_sd_noise_floor = int(np.random.normal(20, 5))
                if irf_sd_noise_floor < 0: irf_sd_noise_floor = 0
                
                
                irf_photoncount = int(np.random.normal(1e7, 1e5))
                
                sim_randomsIRF.append([irf_peak_delay1, irf_peak_delay2, irf_peak_delay3, irf_peak_delay4,
                                        irf_peak_ratio1, irf_peak_ratio2, irf_peak_ratio3, irf_peak_ratio4,
                                        irf_peak_width_ps1, irf_peak_width_ps2, irf_peak_width_ps3, irf_peak_width_ps4,
                                        irf_jitter_std_dev_ps, irf_detector_response_fwhm_ps,
                                        irf_avg_noise_floor, irf_sd_noise_floor, irf_photoncount])
                
                irf, irf_times = irfsim.generate_instrument_function_multi_peak(
                                                    photon_count=irf_photoncount,   
                                                    time_range_ns=irf_timeRange_ns,
                                                    time_resolution_ps=irf_timeResolution_ps,
                                                    peak_widths_ps=[irf_peak_width_ps1, irf_peak_width_ps2, irf_peak_width_ps3, irf_peak_width_ps4],
                                                    jitter_std_dev_ps=irf_jitter_std_dev_ps,
                                                    detector_response_type="gaussian",
                                                    detector_response_params={'fwhm_ps': irf_detector_response_fwhm_ps},
                                                    peak_delays_ps=[irf_peak_delay1, irf_peak_delay2, irf_peak_delay3, irf_peak_delay4],
                                                    peak_ratios=[irf_peak_ratio1, irf_peak_ratio2, irf_peak_ratio3, irf_peak_ratio4],
                                                    avg_noise_floor=irf_avg_noise_floor,
                                                    sd_noise_floor=irf_sd_noise_floor
                                                )
                
                irf = irf - np.min(irf)
                irf = irf / np.sum(irf)
            
            datapoint = sig.convolve(dtof_interpolated, irf, mode='full')[:len(dtof_interpolated)]

            phd_back_cte = np.random.normal(1e-9, 1e-10)
            if phd_back_cte < 0: phd_back_cte = 0
            
            phd_noise_loor_mult = np.random.normal(1, 0.02)
            
            phd_noise_floor_add = np.random.normal(2000, 200)
            if phd_noise_floor_add < 0: phd_noise_floor_add = 0

            phd_nphotons = int(np.random.normal(2e8, 1e6))
            
            sim_randomsPHD.append([phd_back_cte, phd_noise_loor_mult, phd_noise_floor_add, phd_nphotons])

            datapoint = datapoint + phd_back_cte
            datapoint[datapoint < 0] = 0
            probabilities = datapoint / np.sum(datapoint)

            # Generate random times of flight based on the distribution
            bins = np.arange(len(datapoint))
            simulated_arrival_times = np.random.choice(bins, size=phd_nphotons, p=probabilities)

            simulated_phd = np.histogram(simulated_arrival_times, bins=n_channels)[0]
            simulated_mult_noise = np.random.normal(loc=phd_noise_loor_mult, scale=0.02, size=n_channels)
            simulated_add_noise = np.random.poisson(lam=phd_noise_floor_add, size=n_channels)
            simulated_phd = simulated_phd*simulated_mult_noise + simulated_add_noise
            simulated_phd[simulated_phd < 0] = 0
            
            sim_tags.append([ua, ups, rho, iirf])
            sim_irfs.append(irf)
            sim_results.append(simulated_phd)

sim_tags = np.array(sim_tags)
sim_irfs = np.array(sim_irfs)
sim_results = np.array(sim_results)
sim_randomsPHD = np.array(sim_randomsPHD)
sim_randomsIRF = np.array(sim_randomsIRF)

date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
np.save('sim_tags_{}.npy'.format(date_time), sim_tags)
np.save('sim_irfs_{}.npy'.format(date_time), sim_irfs)
np.save('sim_results_{}.npy'.format(date_time), sim_results)
np.save('sim_randomsPHD_{}.npy'.format(date_time), sim_randomsPHD)
np.save('sim_randomsIRF_{}.npy'.format(date_time), sim_randomsIRF)

toc = time.time()
print("Simulation ended. Elapsed time: ", toc - tic, " seconds")