import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import re

from typing import Optional, List, Union

def eyeBehaviour(sam):
    """
    Inputs
    --------
    sam: (tiempo posXOjo posYOjo)
        Matriz de datos por columna.
    
    Outputs
    --------
    tpoFix: ()
    tpoSac: ()
        Vector correspondiente a los tiempos de las sacadas para cada ojo (por ahora solo el ojo derecho) el cual se calcula
        con los siguientes parametros:
        -    39.38 pixeles    = 1 grado visual [�]
        -    amplitud sacada >= 0.1 grados visuales [�]
        -    aceleraci�n     >= 4000 m/s^2
        -    velocidad       >= 30 m/s
        -    tiempo sacada   >= 0.4 ms
           
    Laboratorio de Neurosistemas
    Christ Devia & Samuel Madariaga
    Febrero 2019
    """
    ## PARTE 0: Definici�n de par�metros
    pix2grad       = 39.38
    ampliSaccThld  = 0.1
    acce1Thld      = 4000
    velThld        = 30
    lengthSaccThld = 0.4
    ## PARTE 1: Calcula la velocidad y la aceleracion
    ini = 1
    ter = len(sam)
    nsam = ter-ini+1
    cal = np.zeros((nsam-1, 5)) # [timestamp dist vel acc sacc?]
    for kk in range(nsam-1):
        do = ini+kk-1
        # distancia total recorrida en grados visuales
        DTgr = np.sqrt(  (sam[do+1, 1]-sam[do, 1])**2 + (sam[do+1, 2]-sam[do, 2])**2  ) / pix2grad
        # velocidad en grados/segundos
        vel = DTgr / ((sam[do+1, 0] - sam[do, 0]+1) * 10**-3)
        # aceleracion en grados/seg^2
        acc = vel/((sam[do+1, 0] - sam[do, 0]+1) * 10**-3)
        # Guarda los reusltados
        cal[kk, :4] = [sam[do+1, 0], DTgr, vel, acc]
        # Aceleracion y velocidad
        if (acc >= acce1Thld) or (vel >= velThld):
            cal[kk, 4] = 1 # marca con un 1 los periodos de sacada
    ## PARTE 2: Detecta los tiempos de inicio y termino de sacada
    dife = np.diff(cal[:, 4])
    T1 = np.where(dife > 0)[0] + 1 # Onset of saccades
    T2 = np.where(dife < 0)[0] + 1 # Offset of saccades
    # Hace el match entre el inico y el termino de la primera sacada
    if len(T1) > len(T2):
        T1 = T1[:len(T2)]
        if T1[0] > T2[0]:
            T1 = T1[:-1]
            T2 = T2[1:]
    elif len(T1) < len(T2):
        if T1[0] > T2[0]:
            T2 = T2[1:]
        else:
            T2 = T2[0:-1]
    elif T1[0] > T2[0]:
        T1 = T1[0:-1]
        T2 = T2[1:]
    # Calcula la DURACION de la sacada en samples
    Dsac = T2-T1
    # Verifica que no exitan las sacadas negativas
    if any(Dsac)<0:
        print('Existe una sacada negativa')
        T1[Dsac<0] = []
        T2[Dsac<0] = []
    ## PARTE 3: Solo deja las sacadas con amplitud mayor a 0.1 grados visuales
    # Calcula la amplitud total por sacada
    Asac = np.zeros((len(T1), 1))
    for kk in range(len(T1)):
        Asac[kk] = sum(cal[T1[kk]:T2[kk], 1]) # si hay un NaN dara NaN
    ########################## Ojo 0.5 segun paper de los monkeys
    # Verifica cuales son mayores al umbral de deteccion, en este caso > 0.1
    cond = (Asac > ampliSaccThld).astype(int)
    # Flag de que esa sacada es un blink (por que su amplitud es NaN) y debe dejarlo
    fgbli = np.isnan(Asac).astype(int)
    # Mantiene los que son NaN pues indican que esa sacada es un blink, fuerza
    # a que cond sea 0 solo cuando la amplitud es menor que 0.1
    cond2 = np.zeros((len(cond), 1))
    cond2[np.logical_or(cond==1, fgbli==1)] = 1
    aT1 = T1[cond.squeeze().astype(bool)]
    aT2 = T2[cond.squeeze().astype(bool)]
    # Se generan los tiempos de las fijaciones y la sacadas
    tpoSac = np.array([cal[aT1, 0], cal[aT2, 0]-1, Asac[cond2>0], cal[aT2, 0]-cal[aT1, 0]-1,
              sam[aT1,1], sam[aT1,2], sam[aT2-1,1], sam[aT2-1, 2], np.zeros((len(sam[aT2,2])))]).T
    ## PARTE 4: Con los tiempos de los pesta�eos elimina las sacadas insertas entre estos
    aux = np.zeros((len(tpoSac[:,0])))
    for i in range(1, len(tpoSac)):
        if tpoSac[i, 1] - tpoSac[i, 0] <= lengthSaccThld:
            aux[i] = 1
        else:
            if (tpoSac[i, 0] - tpoSac[i-1, 1] <= 16) and (tpoSac[i, 2] <= 1.5):
                tpoSac[i, 8] = 1
    tpoSac = tpoSac[~aux.astype(bool)]
    aT1 = aT1[~aux.astype(bool)]
    aT2 = aT2[~aux.astype(bool)]
    ## PARTE 5: Con los tiempos de los pesta�eos elimina las sacadas insertas entre estos
    tf1 = aT2[:-1] 
    tf2 = aT1[1:] - 1
    tpoFix = np.array([cal[tf1, 0], cal[tf2, 0], cal[tf2, 0]-cal[tf1, 0], 
              sam[tf1, 1], sam[tf2, 2], np.zeros((len(sam[tf2, 2])))]).T
    aux = np.zeros((len(tpoFix[:, 0])))
    for i in range(len(tpoFix)):
        if tpoFix[i, 1] - tpoFix[i, 0] <= lengthSaccThld:
            aux[i] = 1
    tpoFix = tpoFix[~(aux).astype(bool)]
    return [tpoSac, tpoFix]

def eyeBehaviour_remodnav(data, px2deg=39.38, sampling_rate=500):
    """
    data: dataframe with columns ["x", "y"] and index as a range.
    px2deg:
    sampling_rate:
    """
    from remodnav.clf import EyegazeClassifier
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    clf = EyegazeClassifier(px2deg=px2deg, sampling_rate=sampling_rate)
    pp = clf.preproc(data)
    events = clf(pp, classify_isp=True, sort_events=True)
    return events

def format_output_eyeBehaviour(events, sampling_rate=500):
    tpoSac = events[0]
    tpoFix = events[1]
    # Saccade events
    columns_sac = ["t_i", "t_f", "Asac", "diff_t", "start_x", "start_y", "end_x", "end_y", "blink_flag"]
    df_sac = pd.DataFrame(tpoSac, columns=columns_sac)
    df_sac["duration"] = df_sac["diff_t"]*(1./sampling_rate)
    df_sac_formatted = df_sac[["t_i", "t_f", "start_x", "start_y", "duration"]]
    # Fixation events
    columns_fix = ["t_i", "t_f", "diff_t", "start_x", "start_y", "blink_flag"]
    df_fix = pd.DataFrame(tpoFix, columns=columns_fix)
    df_fix["duration"] = df_fix["diff_t"]*(1./sampling_rate)
    df_fix_formatted = df_fix[["start_x", "start_y", "duration"]]
    return df_sac_formatted, df_fix_formatted

def format_output_remodnav(events):
    df_events = pd.DataFrame(events)
    # Saccade events
    df_sac = df_events[df_events["label"] == "SACC"]
    df_sac["duration"] = df_sac["end_time"] - df_sac["start_time"]
    df_sac_formatted = df_sac[["start_x", "start_y", "duration"]]
    # Fixation events
    df_fix = df_events[df_events["label"] == "FIXA"]
    df_fix["duration"] = df_fix["end_time"] - df_fix["start_time"]
    df_fix_formatted = df_fix[["start_x", "start_y", "duration"]]
    return df_sac_formatted, df_fix_formatted

def detect_sac_fix_from_scanpath(y, fn="eyeBehaviour"):
    if fn == "eyeBehaviour":
        idx = np.expand_dims(np.arange(len(y)), -1)
        y = np.concatenate([idx, y], axis=1)
    eyeBehaviour_fn = eyeBehaviour if fn == "eyeBehaviour" else eyeBehaviour_remodnav
    format_fn = format_output_eyeBehaviour if fn == "eyeBehaviour" else format_output_remodnav
    events = eyeBehaviour_fn(y)
    sac, fix = format_fn(events)
    return sac, fix

def calculate_multimatch(y_real, y_pred, screensize=[1920, 1080],
                         grouping=False,
                         TDir=0.0,
                         TDur=0.0,
                         TAmp=0.0,
                         fn="eyeBehaviour"):
    """
    Calculate multimatch metrics.
    """
    import multimatch_gaze as m
    _, fix_real = detect_sac_fix_from_scanpath(y_real, fn=fn)
    _, fix_pred = detect_sac_fix_from_scanpath(y_pred, fn=fn)
    #return m.docomparison(fix_real, fix_pred, screensize=[1280, 720])
    mm = m.docomparison(fix_real, fix_pred, screensize=screensize)#, grouping=grouping, TDir=TDir, TDur=TDur, TAmp=TAmp)
    return [float(mm_i) for mm_i in mm]

def fit_gaussian(y_real, y_pred):
    """
    Fit Gaussian.
    """
    from scipy.stats import norm as norm_stat

    # Error and norm fitted for x and y
    error_x = y_pred["x"].values - y_real["x"].values
    error_y = y_pred["y"].values - y_real["y"].values
    norm_fitted_x = norm_stat.fit(error_x)
    norm_fitted_y = norm_stat.fit(error_y)
    return norm_fitted_x, norm_fitted_y, error_x, error_y

def detect_spiketrain(data, threshold=2):
    """
    Detect Spikes (Saccades).
    """
    import elephant
    import neo
    import numpy as np
    import quantities as pq
    from elephant.spike_train_generation import peak_detection
    from neo import SpikeTrain
    from neo.core import AnalogSignal
    data_diff = (data[1:]-data[:-1])
    signal = AnalogSignal(data_diff, units='V', sampling_rate=pq.Hz)
    spiketrain_above = peak_detection(signal, threshold=np.array(threshold)*pq.V, sign='above')
    spiketrain_below = peak_detection(signal, threshold=np.array(threshold)*pq.V, sign='below')
    times = np.sort(np.concatenate([spiketrain_above.times, spiketrain_below.times])) * pq.s
    assert spiketrain_above.t_stop == spiketrain_below.t_stop
    spiketrain = SpikeTrain(times=times, t_stop=spiketrain_above.t_stop)
    return spiketrain

def get_cross_correlation_histogram(y_real, y_pred, binsize=5, window=[-30, 30]):
    """
    Calculate Cross Correlation Histogram.
    """
    import quantities as pq
    from elephant.conversion import BinnedSpikeTrain
    from elephant.spike_train_correlation import cross_correlation_histogram
    spike_real = detect_spiketrain(y_real)
    spike_pred = detect_spiketrain(y_pred)
    binned_real = BinnedSpikeTrain(spike_real, bin_size=binsize * pq.s)
    binned_pred = BinnedSpikeTrain(spike_pred, bin_size=binsize * pq.s)
    cc_hist = cross_correlation_histogram(binned_real, binned_pred,
        window=[-30,30],
        border_correction=False,
        binary=False, kernel=None, method='memory')
    max_pos = np.argmax(cc_hist[0].magnitude.squeeze())
    peak = cc_hist[1][max_pos]*binsize
    return cc_hist, int(peak)

def calculate_scanmatch(scanpath1, scanpath2, ScanMatchInfo={}, fn="eyeBehaviour"):
    from utils.ScanMatchCalculator import ScanMatch_FixationToSequence, ScanMatch_Struct
    from utils.ScanMatchCalculator import ScanMatch as scanmatch
    _, fix1 = detect_sac_fix_from_scanpath(scanpath1, fn=fn)
    _, fix2 = detect_sac_fix_from_scanpath(scanpath2, fn=fn)
    try:
        ScanMatchInfo = ScanMatch_Struct(ScanMatchInfo=ScanMatchInfo)
        seq1 = ScanMatch_FixationToSequence(fix1, ScanMatchInfo)
        seq2 = ScanMatch_FixationToSequence(fix2, ScanMatchInfo)
        return scanmatch(seq1=seq1, seq2=seq2, ScanMatchInfo=ScanMatchInfo)
    except Exception as e:
        print("Cant calculate scanmatch, returning nan")
        return np.nan

def calculate_dtw(scanpath1, scanpath2, fn="eyeBehaviour"):
    _, P = detect_sac_fix_from_scanpath(scanpath1, fn=fn)
    _, Q = detect_sac_fix_from_scanpath(scanpath2, fn=fn)
    return DTW(P, Q)

def DTW(P, Q, **kwargs):
    from fastdtw import fastdtw
    dist, _ =  fastdtw(P, Q, dist=euclidean)
    return dist

def calculate_rec(scanpath1, scanpath2, threshold, fn="eyeBehaviour"):
    _, fix1 = detect_sac_fix_from_scanpath(scanpath1, fn=fn)
    _, fix2 = detect_sac_fix_from_scanpath(scanpath2, fn=fn)
    return REC(fix1, fix2, threshold=threshold)

def calculate_det(scanpath1, scanpath2, threshold, fn="eyeBehaviour"):
    _, fix1 = detect_sac_fix_from_scanpath(scanpath1, fn=fn)
    _, fix2 = detect_sac_fix_from_scanpath(scanpath2, fn=fn)
    return DET(fix1, fix2, threshold=threshold)
    

def REC(P,Q, threshold, **kwargs):
	"""
		Cross-recurrence
		https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
	"""
	def _C(P, Q, threshold):
		assert (P.shape == Q.shape)
		shape = P.shape[0]
		c = np.zeros((shape, shape))

		for i in range(shape):
			for j in range(shape):
				if euclidean(P[i], Q[j]) < threshold:
					c[i,j] = 1
		return c
	P = np.array(P, dtype=np.float32)
	Q = np.array(Q, dtype=np.float32)
	min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
	P = P[:min_len,:2]
	Q = Q[:min_len,:2]

	c = _C(P, Q, threshold)
	R = np.triu(c,1).sum()
	return 100 * (2 * R) / (min_len * (min_len - 1))

def DET(P,Q, threshold, **kwargs):
	"""
		https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
	"""
	def _C(P, Q, threshold):
		assert (P.shape == Q.shape)
		shape = P.shape[0]
		c = np.zeros((shape, shape))

		for i in range(shape):
			for j in range(shape):
				if euclidean(P[i], Q[j]) < threshold:
					c[i,j] = 1
		return c
	P = np.array(P, dtype=np.float32)
	Q = np.array(Q, dtype=np.float32)
	min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
	P = P[:min_len,:2]
	Q = Q[:min_len,:2]

	c = _C(P, Q, threshold)
	R = np.triu(c,1).sum()

	counter = 0
	for i in range(1,min_len):
		data = c.diagonal(i)
		data = ''.join([str(item) for item in data])
		counter += len(re.findall('1{2,}', data))


	return 100 * (counter / R)

def LAM(P,Q, threshold, **kwargs):
	"""
		https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
	"""
	def _C(P, Q, threshold):
		assert (P.shape == Q.shape)
		shape = P.shape[0]
		c = np.zeros((shape, shape))

		for i in range(shape):
			for j in range(shape):
				if euclidean(P[i], Q[j]) < threshold:
					c[i,j] = 1
		return c
	P = np.array(P, dtype=np.float32)
	Q = np.array(Q, dtype=np.float32)
	min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
	P = P[:min_len,:2]
	Q = Q[:min_len,:2]

	c = _C(P, Q, threshold)
	R = np.triu(c,1).sum()

	HL = 0
	HV = 0

	for i in range(N):
		data = c[i,:]
		data = ''.join([str(item) for item in data])
		HL += len(re.findall('1{2,}', data))

	for j in range(N):
		data = c[:,j]
		data = ''.join([str(item) for item in data])
		HV += len(re.findall('1{2,}', data))

	return 100 * ((HL + HV) / (2 * R))

def CORM(P,Q, threshold, **kwargs):
	"""
		https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
	"""
	def _C(P, Q, threshold):
		assert (P.shape == Q.shape)
		shape = P.shape[0]
		c = np.zeros((shape, shape))

		for i in range(shape):
			for j in range(shape):
				if euclidean(P[i], Q[j]) < threshold:
					c[i,j] = 1
		return c

	P = np.array(P, dtype=np.float32)
	Q = np.array(Q, dtype=np.float32)
	min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
	P = P[:min_len,:2]
	Q = Q[:min_len,:2]

	c = _C(P, Q, threshold)
	R = np.triu(c,1).sum()

	counter = 0

	for i in range(0, min_len-1):
		for j in range(i+1, min_len):
			couter += (j-i) * c[i,j]

	return 100 * (counter / ((min_len - 1) * R))
