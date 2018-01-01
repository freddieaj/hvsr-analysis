#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from obspy.core import Trace, Stream
from obspy.core import UTCDateTime
from obspy.core import read
from obspy.signal.invsim import cosTaper
import os
from obspy import signal
from numpy import float64
import math
from matplotlib.ticker import ScalarFormatter
from cmath import exp
from transformfunc import transformfunc

#H/V function (This is called later in the script)===================================
def hvfunc(statname,eq_delay,winlen):
        filepathN=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + statname + '_HNN.SAC') #define the filepaths of the time series (north east and vertical)
        filepathE=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + statname + '_HNE.SAC')
        filepathZ=os.path.join('/Users/freddiejackson/Python/catalog5/sacdata/' + statname + '_HNZ.SAC')

        st=read(filepathN) 						#read input files and saves them as stream files
        N=st[0]         						#extracts and saves the first trace from the stream
        st=read(filepathE)
        E=st[0]
        st=read(filepathZ)
        Z=st[0]



        #======Processing==========
        N.detrend('simple') 						#removes simple linear trend
        E.detrend('simple')
        Z.detrend('simple')
        N.detrend('demean') 						#removes mean
        E.detrend('demean')
        Z.detrend('demean')
        N.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True) 	#bandpass filter
        E.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True)
        Z.filter('bandpass', freqmin=0.1, freqmax=20, corners=2, zerophase=True)

        #=======Window parameters==========
        t_start=N.stats.starttime	#extract start time of the trace
        t_end=N.stats.endtime           #extract end time
        trlen=t_end-t_start             #calculates length
        time=np.arange(0,N.stats.npts*N.stats.delta,N.stats.delta)		#creates an array for the time axis of the trace

        eq_t=t_start+eq_delay							#variable for when the S-wave energy arrives

        N_tw1 = N.slice(eq_t, eq_t+winlen)					#slice the traces into a short time window
        E_tw1 = E.slice(eq_t, eq_t+winlen)
        Z_tw1 = Z.slice(eq_t, eq_t+winlen)
        N_tw1.taper(type='cosine',max_percentage=0.1,side='both')		#taper the edges of the time window
        E_tw1.taper(type='cosine',max_percentage=0.1,side='both')
        Z_tw1.taper(type='cosine',max_percentage=0.1,side='both')

        #======Spectral analysis=========
        N_fft=np.fft.fft(N_tw1.data) 			#Fast fourier transform of each trace
	E_fft=np.fft.fft(E_tw1.data)
	Z_fft=np.fft.fft(Z_tw1.data)
	fftn=len(Z_fft) 				#extract no of samples
        df=N.stats.sampling_rate
        print('samp rate is',df)
        #reads the sampling rate from the header info
	N_fft=N_fft[0:fftn/2] 				#remove freqyency content above the nyquist and negative freqs
	E_fft=E_fft[0:fftn/2]
	Z_fft=Z_fft[0:fftn/2]
	freq_axis=np.linspace(0,df/2,fftn/2) 		#defines a frequency axis
	N_freq=abs(N_fft) 				#take amplitude of spectra
	E_freq=abs(E_fft)
	Z_freq=abs(Z_fft)
	freq_axis = float64(freq_axis) 			#change precision
	N_freq = float64(N_freq)
	E_freq = float64(E_freq)
	Z_freq = float64(Z_freq)
							#smooths the spectra
	smoothing_cons=40
	N_freq_smooth=signal.konnoohmachismoothing.konnoOhmachiSmoothing(N_freq, freq_axis, bandwidth=smoothing_cons, max_memory_usage=1024, normalize=False)
	E_freq_smooth=signal.konnoohmachismoothing.konnoOhmachiSmoothing(E_freq, freq_axis, bandwidth=smoothing_cons, max_memory_usage=1024, normalize=False)
	Z_freq_smooth=signal.konnoohmachismoothing.konnoOhmachiSmoothing(Z_freq, freq_axis, bandwidth=smoothing_cons, max_memory_usage=1024, normalize=False)

	N_double=abs(np.power(N_freq_smooth,2)) 	#square components
	E_double=abs(np.power(E_freq_smooth,2))
	H=np.sqrt(np.add(E_double,N_double)/2) 		#take quadratic mean

	HV=np.divide(H,Z_freq_smooth)			#Calculate H/V

        """plt.figure(figsize=(10,7))
        plt.subplot(211)
        plt.semilogx(freq_axis,H,'r-',linewidth=4.0)
        plt.semilogx(freq_axis,Z_freq_smooth,'b-',linewidth=4.0)
        plt.legend(('H','V'))
        plt.subplot(212)
        plt.semilogx(freq_axis,HV,'k-',linewidth=4.0)
        plt.legend(('H/V'))"""

	return time,Z,freq_axis,H,HV;		#return the function outputs

#====================================================================================
#Original HV

stat1='PMB01A1' 				#Names of the earthquake files
stat2='PMB02A2'
stat3='PMB03A3'
tw_len=40					#length of the time window		
eq_delay=291					#delay of S-wave energy

time1,Z1,freq_axis1,H1,HVcurve1=hvfunc(stat1,eq_delay,tw_len)
time2,Z2,freq_axis2,H1,HVcurve2=hvfunc(stat2,eq_delay,tw_len)		#calls function for 3 files
time3,Z3,freq_axis3,H1,HVcurve3=hvfunc(stat3,eq_delay,tw_len)

#plot results
xmin=0.2
xmax=15
ymin=0
ymax=14

plt.figure(figsize=(10,7))
ax=plt.subplot(212)
plt.semilogx(freq_axis1,HVcurve1)
plt.semilogx(freq_axis2,HVcurve2)
plt.semilogx(freq_axis3,HVcurve3)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.xlabel('Frequency (Hz)')
plt.ylabel('H/V')
plt.grid(which='both')

ax.set_title('a) Borehole 1',x=0.5,y=0.8)
plt.legend(('0 m','9 m','41 m'))
plt.savefig('foo.png', bbox_inches='tight')

plt.show()
#==================================================================================
