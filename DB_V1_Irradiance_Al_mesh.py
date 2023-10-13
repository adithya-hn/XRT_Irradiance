#=========
#Improved code of Al_mesh intensity study
# 29/5/22
# > Centre correction bugfix
# > CH small contourese removal
# > 95% radius fix done for variable size
# > Databese file format added
# 4/6/22
# > Overlapping of XBPsis fixxed
# 15/6/22
# Moved to Al5
# CH thresholding is doing with 5% percentile did not worked well- over contribution of CH
# > Segmentation file added
# 20/5/23 Al5 => Al7
# > Log scaled image is set using XRT map function
# > Segmentation file section added
# > Mtx Shift disabled
# > Limb data added
# > Decimal date of observation using Astropy 
# > log file added to have Old DOB, Center coordinates \\ Radius need to be added
# 30/09/2023
# Fits header modification is done
# Al_7_v0 is renamed to this file
#
# 6/10/23 :DB_V0 =>> DB_V1
# > Segmentation file converted bit
# > 
#
#
#=========


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import figure
from astropy.io import fits
from astropy import visualization as av
from astropy.time import Time
import scipy.misc
import math as mt
from jdcal import gcal2jd, jd2gcal
import datetime
import os
import timeit
import cv2
import pathlib
import imageio
import numpy.ma as ms
import scipy.stats as si
from statistics import mode
import zipfile
import sys
import sunpy.map
import astropy.units as u
import shutil
import copy

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import warnings
import gc



def ignore_sunpy_warnings():
    # Filter out specific SunPy warning messages
    warnings.filterwarnings("ignore", category=UserWarning, module="sunpy")
    #warnings.filterwarnings("ignore", category=UserInfo, module="sunpy")

ignore_sunpy_warnings()

#logging.getLogger('sunpy.map').setLevel(logging.WARNING)
#logger.setLevel("info", "DISABLE")
startTime = timeit.default_timer()
totelIm = 0
filetime=0
prvTime=startTime

pathlib.Path("Al_Imgs_DB_V1").mkdir(parents=True, exist_ok=True) #
pathlib.Path("RGNmaps_DB_V1").mkdir(parents=True, exist_ok=True)

newlist = []
GOESdata = []

CHarray = []
BParray = []
ARarray = []
BGarray = []
FDarray = []
Lbarray = []

CHarea = []
BParea = []
ARarea = []
BGarea = []

CHa = []
BPa = []
ARa = []
BGa = []
FDa = []
LBa = []

CHi = []
BPi = []
ARi = []
BGi = []
Fdi = []

n_AR = []
n_BP = []
n_CH = []
l_DOB = []
Ls_DOB = []
l_CHth= []
l_ARth=[]
l_BPth=[]
l_size = []
shapeArry=[]

#Logfiles

d_cx =[]
d_cy =[]
Cx =[]
Cy =[]
Rarray=[]


sf=np.loadtxt('V1_feed_part2_files_Al_Irrad_2008_2023.5.dat',usecols=(0),dtype='str').transpose()
Length = len(sf)
print(Length)

for l in range(Length):
    try:
        img1 = fits.open(sf[l])
        Fname=(os.path.splitext((sf[l].split(os.sep))[-1]))[0]
        scidata1 = img1[0].data
        DOB1 = img1[0].header['DATE_OBS']
        xcen = int(img1[0].header['XCEN']) / (img1[0].header['XSCALE'])
        ycen = int(img1[0].header['YCEN']) / (img1[0].header['YSCALE'])
        size = scidata1.shape
        m = scidata1.mean()

        if size[0] == 1024:
            #print('ok 1k')
            kcanny=canny(scidata1,sigma=2,low_threshold=8,high_threshold=100)
            hough_radii=np.arange(462,478,2)
            hough_res = hough_circle(kcanny, hough_radii)
            accums, cx, cy, rad = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=1)
            Rad_h = int(rad[0])#
            center = (int(cx[0]),int(cy[0]))
            Radrdc=int(Rad_h*0.05)
            Radlb=int(Rad_h*1.05)
            R = int(Rad_h - Radrdc)
            xbpTpix=10
            arTpix=1000
        else:
            kcanny=canny(scidata1,sigma=2,low_threshold=5,high_threshold=30)
            Rad = int(960)
            hough_radii = np.arange(924, 955, 2)
            hough_res = hough_circle(kcanny, hough_radii)
            accums, cx, cy, rad = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
            center = (int(cx[0]),int(cy[0]))
            Rad_h = int(rad[0])#
            Radrdc=int(Rad_h*0.05)
            Radlb=int(Rad_h*1.05)
            center_h = (int(cx[0]), int(cy[0]))
            R = int(Rad_h - Radrdc)
            xbpTpix=10*4
            arTpix=1000*4

    except:
        #start = 0 #if it could not ope the file
        print('**Exception**')
        print('error is:', sys.exc_info()[0], sf[l])
        pass


    
    xrt_map=sunpy.map.sources.XRTMap(scidata1,img1[0].header)
    Filter1 = img1[0].header['EC_FW1_']
    Filter2 = img1[0].header['EC_FW2_']

    start = 1 # Good to go

    if start == 1:
        size = scidata1.shape
        shapeArry.append(size[0])
        
        dob_str = DOB1
        dob_obj = datetime.datetime.strptime(dob_str, '%Y-%m-%dT%H:%M:%S.%f')
        dtm = Time(dob_obj,format='datetime') #date and time
        obsDate=dtm.decimalyear
        l_DOB.append(np.round(obsDate,6))

        circ = np.zeros((size))
        lb   = np.zeros((size))
        CHmask = np.zeros(size, np.uint8)
        BPmask = np.zeros(size, np.uint8)
        BPmask1 = np.zeros(size, np.uint8)
        ARmask = np.zeros(size, np.uint8)
        ARmask1 =np.zeros(size, np.uint8)
        BGmask = np.zeros(size, np.uint8)
        RgnMp = np.zeros(size, np.uint8)

        #AR_cont_cord =np.zeros(size, np.uint8)
        #CH_cont_cord =np.zeros(size, np.uint8)
        #XBP_cont_cord =np.zeros(size, np.uint8)
        
        #==== Limb part =====

        h = int(center[0])
        k = int(center[1])
        cv2.circle(circ, (h, k), R, (255, 0, 0), -1)  # disk
        cv2.circle(lb, (h, k), Radlb, (255, 0, 0), -1)
        cv2.circle(lb, (h, k), R, (0, 0, 0), -1)
        Slb= (np.where(lb,scidata1,0)).sum()
        Bo_LBmask=lb.astype(np.bool_)
        #Lb=Slb.astype(np.uint8)
        #imageio.imwrite('limb.jpg', Lb)

        # ----   ooo      ---

        circle = circ.astype(np.bool_)
        Circle = np.invert(circle)  # hole
        mask = ms.array(scidata1, mask=Circle)
        disk = ms.array(scidata1, mask=circle)  # hided disk
        DD = disk * 0
        dd = ms.array(DD, mask=Circle)

        sun = DD.data  # only solar disk
        index = np.nonzero(sun)
        mtxData = sun[index[0], index[1]]
        s = np.std(mtxData)
        mn = mtxData.mean()
        md = np.median(mtxData)

        threshInt = md * 0.3
        CHt = np.where(sun < threshInt, 255, 0) # CH Thresholding
        CHm = ms.array(CHt, mask=circle)
        CHd = CHm * 0
        CHs = CHd.astype(np.uint8)  # Converting images to 8bit

        # morphological closing
        kernel  = np.ones((15, 15), np.uint8)
        kernel1 = np.ones((5, 5), np.uint8)
        CH = cv2.morphologyEx(CHs, cv2.MORPH_CLOSE, kernel1)

        CHcont, hierarchy = cv2.findContours(CH, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
        CH_cont=[]
        BP_cont_ = []  # for all XBPs
        AR_cont = []
        
        nch=len(CHcont)
        for ch in range(nch):
            ch_a = cv2.contourArea(CHcont[ch])
            if ch_a > 10:
                CH_cont.append(CHcont[ch])
                #ARarea.append(b1)
        
        
        #==== | Scaled images from xrt map | =======
        
        fig = plt.figure()
        ax = plt.subplot(projection=xrt_map)
        xrt_map.plot(cmap='gray',clip_interval=(2, 100)*u.percent)
        ax.set_title('')
        plt.axis('off')
        
        if size[0]==2048:
          fig.set_size_inches(20.48, 20.48)
        else: 
          fig.set_size_inches(10.24, 10.24)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1) 
        fig.canvas.draw()
        plot_matrix = np.array(fig.canvas.renderer.buffer_rgba())
        plot_matrix=(plot_matrix[:,:,:3])[::-1]
        #plt.show(block=False)
        fig.clf()
        plt.close('all')
        # ---- ooo ----

        comb_img=cv2.cvtColor(plot_matrix,cv2.COLOR_RGB2BGR)

        #AR operation
        tests=[]
        ARTh = mn * 2.5
        ARt = np.where(sun > ARTh, 255, 0)
        ARs = ARt.astype(np.uint8)
        AR  = cv2.morphologyEx(ARs, cv2.MORPH_OPEN, kernel1)
        ARcont, hierarchy = cv2.findContours(AR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        nAR = len(ARcont)
        ArX=[]
        ArY=[]
        for j in range(nAR):
            b1 = cv2.contourArea(ARcont[j])
            if b1 > arTpix:
                AR_cont.append(ARcont[j])
                ARarea.append(b1)
                N_arCont=ARcont[j]
                
                for aR in range(len(N_arCont)):
                    xy_cord=N_arCont[aR]
                    #print(xy_cord[0][1])
                    ArX.append(xy_cord[0][0])
                    ArY.append(xy_cord[0][1])
            else:
                BP_cont_.append(ARcont[j])
        
        
        cv2.drawContours(ARmask1,AR_cont, -1, (255, 0, 0), cv2.FILLED)
        AR_masked2_sun = ms.array(sun, mask=ARmask1)
        index2 = np.nonzero(AR_masked2_sun)
        mtxData2 = AR_masked2_sun[index2[0], index2[1]]
        s1 = np.std(mtxData2)
        MeanII = mtxData2.mean()
        _BPTh = MeanII * 1.7  # mn*2.5
        
        #===================
        AR_masked3_sun = np.where(ARmask1,0,sun)
        index3 = np.nonzero(AR_masked3_sun)
        mtxData3 = AR_masked3_sun[index3[0], index3[1]]
        #===================
        
        
        l_BPth.append(_BPTh)
        l_CHth.append(threshInt)
        l_ARth.append(ARTh)
        
        _BPt = np.where(AR_masked2_sun > _BPTh, 255, 0)
        _BPs = _BPt.astype(np.uint8)
        _BP = cv2.morphologyEx(_BPs, cv2.MORPH_CLOSE, kernel)
        _BPcont, hierarchy = cv2.findContours(_BP, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for b in range (len(_BPcont)):
            bp_area= cv2.contourArea(_BPcont[b])
            if bp_area>xbpTpix and bp_area<arTpix:
                BP_cont_.append(_BPcont[b])
        
        cv2.drawContours(BPmask1, BP_cont_, -1, (255, 0, 0), cv2.FILLED)
        BP_cont, hierarchy = cv2.findContours(BPmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(CHmask, CHcont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BPmask, BP_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BGmask, BP_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BGmask, AR_cont, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(BGmask, CH_cont, -1, (255, 0, 0), cv2.FILLED)
        
        cv2.drawContours(comb_img, CH_cont, -1, (255, 0, 255), 2)
        cv2.drawContours(comb_img, BP_cont, -1, (250, 0, 0), 1)
        cv2.drawContours(comb_img, AR_cont, -1, (255, 255, 0), 2)
        cv2.drawContours(comb_img, tests, -1, (0,255, 255), 2)

        Bo_CHmask = CHmask.astype(np.bool_)
        Bo_ARmask1= ARmask1.astype(np.bool_)
        Bo_BPmask = BPmask.astype(np.bool_)
        Bo_BGmask = BGmask.astype(np.bool_)
        In_BGmask = np.invert(Bo_BGmask)
        In_CHmask = np.invert(Bo_CHmask)
        In_ARmask1= np.invert(Bo_ARmask1)
        In_BPmask = np.invert(Bo_BPmask)

        Bo_BGmask_=(BGmask+circle).astype(np.bool_) #Only BG
        
        RgnMp[Bo_BGmask_==True]=2
        RgnMp[Bo_LBmask==True]=4
        RgnMp[Bo_CHmask==True]=8
        RgnMp[Bo_BPmask==True]=16
        RgnMp[Bo_ARmask1==True]=32
        

        #imageio.imwrite('SegMap/{}.jpg'.format(Fname),RgnMp[::-1])

        


        '''
        
        #cv2.circle(RgnMp, (h, k), R, (20, 0, 0), -1)
        #RgnMp=np.where(CHmask,100,RgnMp )
        #cv2.drawContours(RgnMp, BP_cont, -1, (255, 255, 255), cv2.FILLED)
        #cv2.drawContours(RgnMp, AR_cont, -1, (150, 150, 150), cv2.FILLED)
        #cv2.drawContours(RgnMp, CH_cont, -1, (70, 70, 70), cv2.FILLED)
        #cv2.drawContours(RgnMp, CH_cont, -1, (70, 70, 70), cv2.FILLED)

        RgnMp_bp=np.where(BPmask>0,1,0)
        RgnMp_ar=np.where(ARmask1>0,1,0)
        RgnMp_ch=np.where(CHmask>0,1,0)
        RgnMp_bg=np.where(BGmask_>0,1,0)
        RgnMp[RgnMp==255]=1000
        RgnMp[RgnMp==150]=1000000
        RgnMp[RgnMp==70]=100
        RgnMp[RgnMp==20]=10

        RgnMp[RgnMp==255]=3
        RgnMp[RgnMp==150]=4
        RgnMp[RgnMp==70]=2
        RgnMp[RgnMp==20]=1

        RgnMp_=np.unpackbits(RgnMp,axis=-1)
        RGNmp=RgnMp_.reshape(RgnMp.shape +(8,))
        #RgnMp=RgnMp.astype(np.uint4)


        imageio.imwrite('SegMap/{}_bp.jpg'.format(Fname),RgnMp_bp[::-1])
        imageio.imwrite('SegMap/{}_bg.jpg'.format(Fname),RgnMp_bg[::-1])
        imageio.imwrite('SegMap/{}_ar.jpg'.format(Fname),RgnMp_ar[::-1])
        imageio.imwrite('SegMap/{}_ch.jpg'.format(Fname),RgnMp_ch[::-1])

        RgnMp[ARmask==True,0]=1
        RgnMp[BPmask==True,2]=1
        RgnMp[CHmask==True,5]=1
        RgnMp[BGmask==True,6]=1

        RgnMp_bp=(RgnMp[:,:,2]).astype(np.uint8)
        RgnMp_ar=(RgnMp[:,:,0]).astype(np.uint8)
        RgnMp_ch=RgnMp[:,:,5]
        RgnMp_bg=RgnMp[:,:,6]

        print(RgnMp_ar.shape)


        #============:For segmentation file:============== 
        
        cv2.drawContours(AR_cont_cord,AR_cont, -1, (255, 0, 0), 1)
        cv2.drawContours(XBP_cont_cord,BP_cont, -1, (255, 0, 0), 1)
        cv2.drawContours(CH_cont_cord,CH_cont, -1, (255, 0, 0), 1)
        arIn=np.nonzero((AR_cont_cord).transpose()) 
        xbpIn=np.nonzero((XBP_cont_cord).transpose()) 
        chIn=np.nonzero((CH_cont_cord).transpose()) 
        #np.savetxt('ar_conts{}.dat'.format(Fname),np.c_[arIn[0],arIn[1]])
        #np.savetxt('xbp_conts{}.dat'.format(Fname),np.c_[xbpIn[0],xbpIn[1]])
        #np.savetxt('ch_conts{}.dat'.format(Fname),np.c_[chIn[0],chIn[1]])
        
        #==========================
        '''
        

        # masking
        CH_masked_sun = ms.array(sun, mask=In_CHmask)
        BG_masked_sun = ms.array(sun, mask=Bo_BGmask) #mask all fearure
        AR_masked1_sun= ms.array(sun, mask=In_ARmask1)
        BP_masked_sun = ms.array(sun, mask=In_BPmask)
        
        # To avoaid overlapped xbps number
        xbp_Cont, hierarchy= cv2.findContours(BPmask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Numebrs'
        no_of_AR = (len(AR_cont))
        no_of_BP = (len(xbp_Cont)) #total xbps, including ar excluded xbps
        no_of_CH = len(CH_cont)
        n_AR.append(no_of_AR)
        n_BP.append(no_of_BP)
        n_CH.append(no_of_CH)

        # Totel area
        cha = np.count_nonzero(CHmask)
        bpa = np.count_nonzero(BPmask)
        ara = np.count_nonzero(ARmask1)

        A4 = np.count_nonzero(In_BGmask)  # Bg + oudside disk
        A5 = np.count_nonzero(sun)  # fuldisk size
        A6 = size[0] * size[1]  # whole image
        A7 = A6 - A5  # outside disk
        A8 = A4 - A7  # Pure BG

        if size[0] == 2048:
            cha = cha/ 4
            bpa = bpa/ 4
            ara = ara/ 4
            A8  = A8 / 4
            A5  = A5 /4
        
        limbA= (np.pi *Radlb*Radlb)-(np.pi *R*R)

        CHa.append(cha)
        BPa.append(bpa)
        ARa.append(ara)
        BGa.append(A8)
        FDa.append(A5)
        LBa.append(limbA)

        # Totel Intensity
        bpsum = BP_masked_sun.sum()
        csum  = CH_masked_sun.sum()
        bgsum = BG_masked_sun.sum()
        Arsum = AR_masked1_sun.sum()
        tsum  = sun.sum()

        if no_of_BP == 0:
            bgsum=0
        if no_of_CH== 0:
            csum=0
        if no_of_AR == 0:
            Arsum=0
        Totel = csum + bgsum + bpsum + Arsum
        XBP = bpsum  ##

        CHarray.append(csum)
        BParray.append(XBP)
        ARarray.append(Arsum)
        BGarray.append(bgsum)
        FDarray.append(tsum)
        Lbarray.append(Slb)

        ch_p= (((csum) / Totel) *100)
        bp_p= (((XBP) / Totel) * 100)
        ar_p= (((Arsum) / Totel) * 100)
        bg_p= (((bgsum) / Totel) * 100)
        #print('%',ch_p,ar_p)
        CHi.append(ch_p)
        BPi.append(bp_p)
        ARi.append(ar_p)
        BGi.append(bg_p)
        Ls_DOB.append(Fname)

        #=== | Drawing Fit, Limb and 95% Rad circles |  =======
        color1 = (255, 165, 0)  #Orange
        color2 = (64, 254, 208) #Blue
        cv2.circle(comb_img,(h,k), R, color2, 1)       # 95% Rad
        cv2.circle(comb_img,(h,k), Rad_h, color1, 1)   # Actual Rad
        cv2.circle(comb_img,(h,k), (Radlb), color2, 1) # Limb 1.05 Rad
        #print(Rad_h,Radlb)
        # ------ ooo -----

        #========| Segmentation file with Header information |============

        data=fits.PrimaryHDU(RgnMp)
        ct=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        xcorr=int(int(img1[0].header['CRPIX1'])-h)*img1[0].header['XSCALE']
        ycorr=int((int(img1[0].header['CRPIX2'])-k))*img1[0].header['YSCALE']
        dlen=len(img1[0].header)-1
        key_list=list((img1[0].header).keys()) #Keyword list
        hist_pos=key_list.index('HISTORY')
        data.header['DATE']=  ct #datetime.datetime.now() #img1[0].header['DATE']
        data.header['DATE_OBS']=img1[0].header['DATE_OBS']
        data.header['TIME-OBS']=img1[0].header['TIME-OBS'] 
        data.header['FILEORIG']=((Fname+'.fits'),'XRT al_mesh image file')
        data.header['ORIGIN']= 'H.N. Adithya'
        data.header['CONTRIB']= 'Kariyappa et. al'
        data.header['PROG_VER']= 'DB_V1.00_Irrad_Al_Oct_23'
        data.header['CRPIX1']= img1[0].header['CRPIX1']
        data.header['CRPIX2']= img1[0].header['CRPIX2']
        data.header['CRVAL1']= img1[0].header['CRVAL1']
        data.header['CRVAL2']= img1[0].header['CRVAL2']
        data.header['CDELT1']= img1[0].header['CDELT1']
        data.header['CDELT2']= img1[0].header['CDELT2']
        data.header['CUNIT1']= img1[0].header['CUNIT1']
        data.header['CUNIT2']= img1[0].header['CUNIT2']
        data.header['CTYPE1']= (img1[0].header['CTYPE1'], 'Provided by XRT')
        data.header['CTYPE2']= (img1[0].header['CTYPE2'], 'Provided by XRT')
        data.header['RSUN_OBS']= (Rad_h*img1[0].header['YSCALE'],'By circle fit')
        data.header['SAT_ROT']= img1[0].header['SAT_ROT']
        data.header['INST_ROT']= img1[0].header['INST_ROT']
        data.header['CROTA1']= img1[0].header['CROTA1']
        data.header['CROTA2']= img1[0].header['CROTA2']
        #data.header['CROTA1']= img1[0].header['CROTA1']
        data.header['XCEN']= (xcorr,'By circle fit' )
        data.header['YCEN']= (ycorr,'By circle fit' )
        data.header['XSCALE']= img1[0].header['XSCALE']
        data.header['YSCALE']= img1[0].header['YSCALE']
        data.header['FOVX']= img1[0].header['FOVX']
        data.header['FOVY']= img1[0].header['FOVY']
        data.header['PLATESCL']= img1[0].header['PLATESCL']
        data.header['BYTECNT']= img1[0].header['BYTECNT']
        data.header['PIXCNT']= img1[0].header['PIXCNT']
        data.header['BITSPP']= img1[0].header['BITSPP']
        data.header['HISTORY']=img1[0].header[hist_pos:(dlen+1)] #All history of Original files are copied
        del data.header['EXTEND']

        data.writeto("{}_seg.fits".format(Fname),overwrite=True)
        zip_file = zipfile.ZipFile("{}_seg.fits.zip".format(Fname), 'w')
        zip_file.write("{}_seg.fits".format(Fname), compress_type=zipfile.ZIP_DEFLATED) #Zipping the file to decrease the size
        zip_file.close()
        os.replace("{}_seg.fits.zip".format(Fname), os.getcwd()+'/RGNmaps_DB_V1/{}_seg.fits.zip'.format(Fname)) #location of zipped file
        os.remove("{}_seg.fits".format(Fname)) #removing the source file after zipping

        #------ ooo ----------
        #-Log files
        d_cx.append(h) #Centre from circle fitting
        d_cy.append(k)
        Cx.append(xcen)#Centre from Source file
        Cy.append(ycen)
        Rarray.append(Rad_h)

        imageio.imwrite('Al_Imgs_DB_V1/{}.jpg'.format(Fname), comb_img[::-1]) #Image

        f = open('DB_V1_X-ray_data.dat', 'a')
        np.savetxt('DB_V1_X-ray_data.dat', np.c_[
            CHarray, BParray, ARarray, BGarray,Lbarray, CHi, BPi, ARi, BGi, n_BP, n_AR, n_CH, l_DOB, CHa, BPa, ARa, BGa,LBa, FDarray, FDa, shapeArry],
                   fmt='%11.6f',
                   header=' CH Int,   XBP Int,     AR Int,  Background,   Limbint | %intensity-CH   XBP          AR         BG		nXBP		nAR		nCH		l_DOB	    CHa       BPa       ARa        BGa   LBa   FD-int    FDa       Shape')

        f.close()
        
        fa1 = open('V1_Irradianc_Database_Al_mesh.dat', 'a') #area, bg and xbp
        np.savetxt('V1_Irradianc_Database_Al_mesh.dat', np.c_[Ls_DOB,l_DOB,FDarray,ARarray,BParray,CHarray,BGarray,Lbarray,ARi, BPi, CHi, BGi, n_AR, n_BP, n_CH,ARa, BPa, CHa, BGa,FDa,LBa,shapeArry], fmt='%s',
                   header=' File name                    DOB | FDI | ARint | XBPint | CHint | BGint | Limbint  | AR% | XBP% | CH% | BG% | nAR | nXBP | nCH | ARa | XBPa | CHa | BGa | FDa | Shape ')
        fa1.close()

        fb = open('Almesh_log.dat', 'a') #area, bg and xbp
        np.savetxt('Almesh_log.dat', np.c_[Ls_DOB,l_DOB,d_cx,d_cy,Cx,Cy, Rarray,l_ARth,l_CHth,l_BPth,], fmt='%s',
                   header=' File name                    DOB | cx | cy| xcen| ycen Rarray, AR_thresh, CH_thresh, BP_thresh')
        fb.close()

        #= Program run time per image 
        tempstopTime = timeit.default_timer()
        filetime = np.round((tempstopTime - prvTime),2)
        prvTime=tempstopTime
        print('[',l+1,'/', Length,']',DOB1,'|  TIME', filetime)
        # ------ ooo  -------#



stopTime = timeit.default_timer()
runtime = (stopTime - startTime)
TotTime=runtime/3600 #in Hours

print('')
print('......COMPLEETED.....')
print('Time taken', TotTime,'Avg. Time per image', runtime/Length)
print('--------------------------------------------------')


#import Ti_poly





