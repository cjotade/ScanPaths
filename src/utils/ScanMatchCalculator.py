import numpy as np

def ScanMatch_GridMask(Xres, Yres, binX, binY):
    from scipy.interpolate import interp2d

    # resize the mask to be at the right output size without using 'imresize'
    m = binX / Xres
    n = binY / Yres

    xi = np.arange(binX)
    yi = np.arange(binY)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = np.reshape(np.arange(binX*binY), (binY, binX))

    xn = np.floor(np.arange(0, binX, m))
    yn = np.floor(np.arange(0, binY, n))

    ip2d = interp2d(Xi, Yi, Zi); 
    Zn = ip2d(xn, yn)
    #return Zn
    return Zn + 1

def ScanMatch_CreateSubMatrix(Xbins, Ybins, threshold):
    mat = np.zeros((Xbins * Ybins, Xbins * Ybins))
    idx_i = 0
    idx_j = 0

    for i in range(Ybins):
        for j in range(Xbins):
            for ii in range(Ybins):
                for jj in range(Xbins):
                    mat[idx_i, idx_j] = np.sqrt((j-jj)**2 + (i-ii)**2)
                    idx_i +=1
            idx_i = 0
            idx_j += 1
    max_sub = mat.max()
    return np.abs(mat - max_sub) - (max_sub - threshold)

def ScanMatch_Struct(ScanMatchInfo={}):
    def _check_ScanMatch_Structure(ScanMatchInfo):
        check_keys = [
            'Xres', 'Yres', 'Xbin', 'Ybin', 
            'RoiModulus', 'Threshold', 'mask', 'SubMatrix', 'GapValue', 'TempBin'
        ]
        for key in check_keys:
            if not (key in ScanMatchInfo.keys()):
                return False
        return True
    
    if not _check_ScanMatch_Structure(ScanMatchInfo):
        ScanMatchInfo = {
            "Xres": 1920,
            "Yres": 1080,
            "Xbin": 12,
            "Ybin": 8,
            "RoiModulus": 12,
            "Threshold": 3.5,
            "GapValue": 0,
            "TempBin": 0.1
        }
        ScanMatchInfo["SubMatrix"] = ScanMatch_CreateSubMatrix(
            ScanMatchInfo["Xbin"], 
            ScanMatchInfo["Ybin"], 
            ScanMatchInfo["Threshold"]
        )
        ScanMatchInfo["mask"] = ScanMatch_GridMask(
            ScanMatchInfo["Xres"], 
            ScanMatchInfo["Yres"], 
            ScanMatchInfo["Xbin"], 
            ScanMatchInfo["Ybin"]
        )
    return ScanMatchInfo

def ScanMatch_FixationToSequence(scanpath, ScanMatchInfo):
    from utils.utils import round_parser
    scanpath = np.array(scanpath, dtype=np.float32)
    # Any negative value will be set to 1
    scanpath[:, 0][scanpath[:, 0] < 0] = 1
    scanpath[:, 1][scanpath[:, 1] < 0] = 1

    # Fixations outside the screen resolution will be set to the screen resolution
    scanpath[:, 0][scanpath[:, 0] > ScanMatchInfo["Xres"]] = ScanMatchInfo["Xres"] - 1
    scanpath[:, 1][scanpath[:, 1] > ScanMatchInfo["Yres"]] = ScanMatchInfo["Yres"] - 1

    # ---- Get eye movement sequences ----
    def _select_from_mask(scanpath, mask):
        scanpath = scanpath.astype(np.int64)
        subs = np.ravel_multi_index([scanpath[:, 1], scanpath[:, 0]], mask.shape, order="F") 
        sel_subs = np.unravel_index(subs, shape=mask.shape, order="F")
        seq_num = mask[sel_subs]
        return round_parser(seq_num)

    seq_num = _select_from_mask(scanpath, ScanMatchInfo["mask"])

    # ---- Temporal binning if needed ----
    if ScanMatchInfo["TempBin"] != 0:
        # Check if fixation times are available
        assert scanpath.shape[-1] == 3 # fixation times available
        fix_time = round_parser(scanpath[:, 2] / ScanMatchInfo["TempBin"])
        seq = []
        for i in range(1, scanpath.shape[0]):
            seq_add = (seq_num[i] * np.ones(fix_time[i])).tolist()
            seq = seq + seq_add
        seq_num = seq
    # ---- Create the string sequences ----
    seq_str = ScanMatch_NumToDoubleStr(seq_num, ScanMatchInfo["RoiModulus"])
    
    return seq_str

def ScanMatch_NumToDoubleStr(seq_num, modulus):
    # ---- Start conversion ----
    seq_str = ""
    for num in seq_num:
        seq_str += chr((int((num-1)/modulus)) + 65).lower()
        seq_str += chr(int((num-1) % modulus) + 65)
    return seq_str

def ScanMatch_DoubleStrToNum(string, modulus):
    # Upper case the string
    string = string.upper()
    # lengh of string
    str_l = len(string)
    
    # ---- do the processing two by two letters ----
    res = []
    for i in range(0, str_l, 2):
        doubleStr = string[i:i+2]
        num_str = np.array([ord(doubleStr[0]), ord(doubleStr[1])]) - 64
        res_i = ((num_str[0] - 1) * modulus + num_str[1] - 1)
        res.append(res_i)
        
    # Add one to the final array as the conversion works from zero
    res = np.array(res) + 1
    return res


def ScanMatch_nwAlgo(intseq1, intseq2, SubMatrix=None, gap=0):
    m = len(intseq1)
    n = len(intseq2)
    # set up storage for dynamic programming matrix
    F = np.zeros((n+1, m+1))
    F[1:, 0] = gap * np.arange(n).T
    F[0, 1:] = gap * np.arange(m)
    
    # and for the back tracing matrix
    pointer = 4 * np.ones((n+1, m+1), dtype=int)
    pointer[:, 0] = 2
    pointer[0, 0] = 1
    
    # initialize buffers to the first column
    ptr = pointer[:, 1] # ptr(1) is always 4
    currentFColumn = F[:, 0]

    # main loop runs through the matrix looking for maximal scores
    for outer in range(1, m+1):
        # score current column
        scoredMatchColumn = SubMatrix[intseq2-1, intseq1[outer-1]-1]
        # grab the data from the matrices and initialize some values
        lastFColumn = currentFColumn
        currentFColumn = F[:, outer]
        best = currentFColumn[0]
        
        for inner in range(1, n+1):
            # score the three options
            up = best + gap # insert
            left = lastFColumn[inner] + gap # delete
            diagonal = lastFColumn[inner-1] + scoredMatchColumn[inner-1] # Match
            
            # max could be used here but it is quicker to use if statements
            if up > left:
                best = up
                pos = 2
            else:
                best = left
                pos = 4
                
            if diagonal >= best:
                best = diagonal
                ptr[inner] = 1
            else:
                ptr[inner] = pos    
            currentFColumn[inner] = best
        # put back updated columns  
        F[:, outer] = currentFColumn
        # save columns of pointers
        pointer[:, outer] = ptr
    """
    # Find the best route throught the scoring matrix
    i, j = n+1, m+1
    path = np.zeros(n+m, 2)
    step = 0
    
    while (i > 1 or j > 1):
        p = pointer[i, j]
        if p == 1:
            i -= 1
            j -= 1
            path[step, :] = [j,i]
        elif p == 2:
            i -= 1
            path[step, 1] = i
        elif p == 4:
            j -= 1
            path[step, 0] = j
        elif p == 6:
            j -= 1
            path[step, 0] = j
        else:
            j -= 1
            i -= 1
            path[step, :] = [j, i]
        step += 1
    """
    score = F[n, m] 
    return score

def ScanMatch(seq1, seq2, ScanMatchInfo={}):

    ScanMatchInfo = ScanMatch_Struct(ScanMatchInfo)
    
    # use numerical arrays for easy indexing
    intseq1 = ScanMatch_DoubleStrToNum(seq1, ScanMatchInfo["RoiModulus"])
    intseq2 = ScanMatch_DoubleStrToNum(seq2, ScanMatchInfo["RoiModulus"])
    
    # Perform the Needleman-Wunsch alogorithm
    score = ScanMatch_nwAlgo(intseq1, intseq2, ScanMatchInfo["SubMatrix"], ScanMatchInfo["GapValue"])
    
    # Normalise output score
    max_sub = ScanMatchInfo["SubMatrix"].max()
    scale =  max_sub * max(len(intseq1), len(intseq2))
    score = score / scale 

    return score