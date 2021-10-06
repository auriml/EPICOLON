#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os


# In[11]:


import torch
from fastai.vision.all import *
from fastai.distributed import *


# In[12]:


import fastai
import sklearn.metrics as skm
import pandas as pd


# In[15]:


import numpy as np
from matplotlib import pyplot as plt
import openslide
from numpy import asarray
import cv2


# In[16]:


#to be used for WSI images (not TMAs)
class WSI:
    def __init__(self, svs_fn):
        self.svs_fn = svs_fn
        
        
    @staticmethod
    def get_tile( x,y , size= (300,300) , level = 0, path = None, svs_fn = None):

        slide = None

        try:
            slide = openslide.OpenSlide(os.path.join(path, svs_fn + '.svs'))
            #print(f'reading: {tma}')
        except: 
            print(f'failed: {os.path.join(path, svs_fn + ".svs")}')
            return np.nan


        img = array(slide.read_region((x,y),level,size))


        if (len(img.shape) > 2 and img.shape[2] > 3):
            img = img[:,:,:3] #remove last image dimension, so to have 3 channels and not 4 (hue)

        #if Spot.check_scarce_tissue(image):
        #    return np.nan
        return img


# In[7]:


class TMA:
    diameter_spot = 1200
    def __init__(self, tma_path, file, img_path_tif):
        self.meta_fn = file
        self.meta_path = tma_path
        self.img_path_tif = img_path_tif
        if 'Series' in file: #v2 Qpath
            self.name = file.split(' - ')[1].strip('.tif')
        else:
            self.name = file.strip('.txt')[len('TMA results - '):]
        #self.spots = None #self.load_spots_meta()
        #self.img_tif = None #self.load_img_tif()

    @property
    def img_tif(self):
        self.__img_tif = self.load_img_tif()
        return self.__img_tif
    
    @property
    def spots(self):
        self.__spots = self.load_spots_meta()
        return self.__spots
        
    def load_tma_meta(self):
        df1 = None
        try:
            df1 = pd.read_csv(os.path.join(self.meta_path, self.meta_fn), names=['fn', 'missing','X', 'Y', 'ID', 'notes'], sep='\t', skiprows=[0])
            
            if 'tif' in df1.fn[0]: #Qpath v2 Image	Name	Missing	Centroid X µm	Centroid Y µm	Unique ID
                df1 = pd.read_csv(os.path.join(self.meta_path, self.meta_fn), names=['TMA', 'fn', 'missing','X', 'Y', 'ID'], sep='\t', skiprows=[0])
           
                
        except:
            print(os.path.join(self.meta_path, self.meta_fn))
        return df1
    
    def load_img_tif(self):
        a = None
        try:
            #print(f"opening tif: {os.path.join(self.img_path_tif, self.name + '.tif')}")
            a = openslide.OpenSlide(os.path.join(self.img_path_tif, self.name + '.tif'))
        except: 
            print(f"BAD {self.name}.tif")
        return a
    
    def load_spots_meta(self):
        spots = []
        for index, rows in self.load_tma_meta().iterrows():
            spots.append(Spot(self.diameter_spot, 
                              rows.fn, rows.missing, (rows.X, rows.Y), str(rows.ID),
                        self.meta_path, self))
        return spots


# In[8]:


class Spot: 
    def __init__(self, diameter, name, missing, center, ID, img_path_jpg, tma):
        self.path_jpg = img_path_jpg
        self.diameter = diameter #in micrometers
        self.name = name
        self.center = center #in micrometers
        self.missing = missing
        self.ID = ID
        self.TMA = tma
        self.tiles = []
        
    def load_img_jpg(self):
        s = openslide.ImageSlide(os.path.join(self.path_jpg, self.name + '.jpg'))
        #regions to be read with method s.read_region()
        return s
    
    #get spot center coordinates in pixels
    def get_center_pixel(self):
        tma = self.TMA.img_tif
        c_x, c_y = self.center 
        mpp_x, mpp_y = float(tma.properties['openslide.mpp-x']), float(tma.properties['openslide.mpp-y']) #mpp  - 0.25  micras per pixel
        pc_x, pc_y = int(c_x / float(mpp_x)),int(c_y / float(mpp_y))
        return pc_x, pc_y
    
    #return radious in pixels both at full magnification (level = 0) and at a given zoom-out level
    def get_radious_pixel(self, level):
        tma = self.TMA.img_tif
        mpp_x, mpp_y = float(tma.properties['openslide.mpp-x']), float(tma.properties['openslide.mpp-y'])
        #mpp = (mpp_x**2 + mpp_y**2)**0.5
        pspot_radio = int(self.diameter/ mpp_x / 2 )
        pzoom_radio = int( pspot_radio/ 2**level ) #with each increasing level the image is reduced half size 
        return pspot_radio, pzoom_radio
    
    #return spot circunference's north and sud y-coordinates (absolute positions in level 0) 
    #given a relative position from 0 (left) to 1 (right) in the spot diameter in axis x
    def get_circle_coordinates_pixel(self, rel_pos_diameter_x = .5): #return by default the y-coordinates for x fixed at center
        pc_x, pc_y = self.get_center_pixel()
        pspot_radio, _ = self.get_radious_pixel(0)
        x = int(pc_x - pspot_radio + rel_pos_diameter_x*(pspot_radio*2))
        if (( x + pspot_radio < pc_x) or ( x - pspot_radio > pc_x)): #x should be in circle diameter (pc_x - pspot_radio, pc_x + pspot_radio)
            return x, np.nan, np.nan
        root = (pspot_radio**2 - (x - pc_x)**2)**0.5
        y_north = int( -root + pc_y)
        y_sud = int( root + pc_y)
        return x, y_north, y_sud
    
    #check (true or false) if there is scarce tissue in an image
    @staticmethod
    def check_scarce_tissue(tile_image): 
        s = np.mean(tile_image, axis = 2) #collapse 3 color channels into 1 by mean
        s = s[(s > 0)] #remove black mask (value == 0)
          
        return s.mean() >= 230 #completely white = 255
       
    
    #return an image tile from the spot where x,y is the left-sup corner of the tile
    #for reference for current scanned tifs properties, level 0 is fully magnified image and for level 4 encloses one spot of diameter 1.2 micrometers
    def get_tile(self, x,y , size=(300,300) , level = 4, path = None, tif_name = None):
        
        tma = None
        if (tif_name == None): tma = self.TMA.img_tif
        else:
            try:
                tma = openslide.OpenSlide(os.path.join(path, tif_name + '.tif'))
            except: 
                print(tif_name + '.tif')
        
        img = array(tma.read_region((x,y),level,size))
        
        
        if (len(img.shape) > 2 and img.shape[2] > 3):
            img = img[:,:,:3] #remove last image dimension, so to have 3 channels and not 4 (hue)
        
        #if Spot.check_scarce_tissue(image):
        #    return np.nan
        return img

   
        
    
    #check (true or false) that tile (defined by left sup coordinate and size in a zoom level) is enclosed in a box defined by the 4 spot poles 
    def check_tile_in_spot(self, x,y,size=(300,300) , level = 4):
        spot_poles = (self.get_circle_coordinates_pixel(0)[0], #left
                  self.get_circle_coordinates_pixel(1)[0],  #right
                  self.get_circle_coordinates_pixel(0.5)[1], #north
                  self.get_circle_coordinates_pixel(0.5)[2]) #sud
        if (x < spot_poles[0]) or (y < spot_poles[2] ) or (x + size[0]*(2**level) > spot_poles[1] ) or (y + size[1]*(2**level) > spot_poles[3]):
            return False, spot_poles
        else:
            return True, spot_poles
        
        
    #given a tile size in pixels and zoom level then sample all possible tiles from spot 
    #with a given overlap (0 to <1)
    #return all tiles (defined by sup-lef corner coordinates, size and level) 
    #enclosed in box-spot with enough tissue 
    def sample_tiles(self, size = (300,300), level = 4, overlap = .2):
        self.tiles = []
        pspot_radio, _ = self.get_radious_pixel(level)
        spot_tile_ratio = pspot_radio * 2 / size[0] 
        spot_tile_ratio_by_zoom = spot_tile_ratio / 2**level 
        stride_norm = 1 - overlap
        #print(np.arange(0,1,stride_norm/spot_tile_ratio_by_zoom))
        for i_x in np.arange(0,1,stride_norm/spot_tile_ratio_by_zoom): 
            x, y_north, y_sud = self.get_circle_coordinates_pixel(i_x)
            if (y_sud-y_north == 0): #intersection of ecuador with circunference
                if self.check_tile_in_spot(x, int(y_north), size, level)[0]:
                    self.tiles.append([x, int(y_north), size, level])
                if self.check_tile_in_spot(x, int(y_north) - pspot_radio, size, level)[0]:
                    self.tiles.append([x, int(y_north) - pspot_radio, size, level])
            else:
                for j in np.arange(y_north, y_sud, stride_norm *size[1]*(2**level) ): 
                    if self.check_tile_in_spot(x, int(j), size, level)[0]:
                        self.tiles.append([x, int(j), size, level])

        return self.tiles
    
    

    #given a zoom level it returns the box enclosing only this spot
    def get_enclosing_box(self, level, return_image = True):
        pspot_radio, pzoom_radio = self.get_radious_pixel(level)
        #print(pzoom_radio)
        pc_x, pc_y = self.get_center_pixel()
        x,y,size,level = pc_x - pspot_radio, pc_y - pspot_radio,  (pzoom_radio * 2, pzoom_radio * 2), level
        if return_image: 
            tile = self.get_tile(x,y,size,level )
        else: 
            tile = [x,y,size,level]
        return tile
    
    #given a tile defined by [x,y,size,zoom_level] return an enlarged tile 
    #the enlarged tile is defined as the minimal tile needed to enclose original tile rotated around the tile center by n_degrees (values 0 to 90º)
    #for default rotation (45º) the enlargement factor is hence equal to hypothenuse of a side of length 1 triangle (1/sin 45º = 2^(1/2))
    @staticmethod
    def enlarge_tile(x,y, original_tile_size = (300,300), level = 0, n_degrees = 45 ):
        #calculate the enlargement factor for rotation degrees
        enlargement_factor = np.sin(np.deg2rad(n_degrees + 45))/np.sin(np.deg2rad(45))
        #calculate half side length of original tile in pixels in level 0 (max zoom-in or pixels in source image)
        half_side_length = (2**level) * (original_tile_size[0]/ 2)  #it is assuming tile sizes are always square
        #calculate enlarged side length of new enclosing tile to include original tile rotated n degrees
        new_half_side_length = half_side_length * enlargement_factor
        #calculate new upper-left corner coordinate (new_half_side_lenghth need to be substracted because origin of image in pixels is always upper-left image corner)
        x_new,y_new = int(x + (half_side_length - new_half_side_length)), int(y + (half_side_length - new_half_side_length) )
        return [x_new,y_new,(int(original_tile_size[0] * enlargement_factor), int( original_tile_size[0] * enlargement_factor) ),level]
        
    
    #Given an image, define the maximum circle enclosed in the image 
    #and return the image with areas outside the circle masked in black (value 0)
    @staticmethod
    def mask_outside_of_enclosed_cyrcle(img):
        
        hh, ww = img.shape[:2]
        hh2 = hh // 2
        ww2 = ww // 2

        # define circle
        radius = hh2
        xc = hh2
        yc = ww2



        # draw filled circle in white on black background as mask
        img = np.array(img)
        if (img.shape[2] > 3):
            img = img[:,:,:3] #remove last image dimension, so to have 3 channels and not 4 (hue)
        mask = np.zeros_like(img)
        mask = cv2.circle(mask, (xc,yc), radius, (255,255,255), -1)

        # apply mask to image
        result = cv2.bitwise_and(img, mask)

        
        return result


# In[9]:


#OLD ddbb
path = 'SPOTS/TMA06-03.qptma.data/'
#df_old= pd.read_csv('DTS2020 ALENDA.xlsx - DTS2020_ALENDA.csv', )
df_old = pd.read_excel('DTS2020 ALENDA.xlsx', )
df_old_labels = df_old[['COD_DTS','Epicolon1+2IHQ-IMS-maria paper lynlike_IMS','LynchIMS','ihq_mlh1','dukes_r',
 'TNMagrup','BaseEP1y2actualizada2016-OSCAR-def_n', 'grado_di','infirec','moc',
'edat', 'sexe','ccr_sin','aden_sin','r_beth_4',
'RECIMORT','KRAS','localizacion','fechreci','ILEact','estfseagrup']]
pd.set_option('display.max_colwidth', None)
#print(df_old.head())
df_old.columns.values


# In[10]:


df_old[(df_old['NºEpicolon'].isna()) & (df_old['BaseEP1y2actualizada2016-OSCAR-def_ID_EPI2_PRY'].isna()) & (df_old['COD_DTS'].isna()) ]
df_old['mixID'] = df_old.apply(lambda x: {x['NºEpicolon'], x['BaseEP1y2actualizada2016-OSCAR-def_ID_EPI2_PRY'], x['COD_DTS']}, axis = 1)
df_old['mixID'] 


# In[11]:


import re
def f(x):
    x = [x] if not isinstance(x, list) else x
    
    found = set()
    for s in x:
        
        m = re.search('([1-9]\d+)', str(s))
        if m:
            found.add(m.group(1))
    
    return list(found)[0] 


# In[12]:



df_old['new_id'] = df_old['mixID'].apply(lambda x: f(x))
#df_old.new_id.nunique()


# In[35]:



df = pd.read_csv('DTS2020_08_10.csv', )
#add additional clinical variables from old database 
df = df.merge(df_old_labels.loc[df_old_labels.COD_DTS.isna() == False], on = 'COD_DTS', how = 'left')
df.to_csv('cDTS2020_08_10.csv', index = False)



df_labels = df
#print(df_labels.columns)

#print(df_labels.shape, df_old.shape)

QuProject_path = "SPOTS/save-jul20"


# In[37]:


#Load only clinical info of New TMAs provided on March 21 and project with new TMAs
new_TMAs = True
if new_TMAs:
    df_marc21 = pd.read_csv('NUEVOS_HGUA_Mar21.csv')
    df_marc21['LIST_ID'] = df_marc21.BIOPSIA
    df_marc21['patient_ID'] = df_marc21.BIOPSIA
    df_marc21['label'] = df_marc21.MMR
    df_marc21['COD_DTS'] = df_marc21.BIOPSIA

    df_labels = df_marc21
    QuProject_path = "SPOTS/save-mar21"


# # List available spots
# 

# In[31]:


import os
tmas = []
spots = []
for root, dirs, files in os.walk(QuProject_path):
    for file in files:
        if file.endswith(".txt") and re.search('.*', file) :  #To process all tif files replace regex by .*
            tma = TMA(root, file, 'DTS2019')
            spots.append([s.ID for s in tma.spots])
            tmas.append(tma)
            

spots = [i for l in spots for i in l]  

sp = set(spots)


# In[33]:


#set(df_old.COD_DTS) & set(df.COD_DTS)


# # Generate Dataset 

# In[20]:



ds = pd.DataFrame(columns=['tile', 'tile_rot_augmentation', 'path','tif_fn', 'patient_ID','spot_ID','label'])


# In[1]:


def process_spot(spot,label = 'MMR', size=(300,300) ,level=3, verbose = False, overlap = .0, skip_scarce_tissue = False):
    m = ''
    ds = pd.DataFrame(columns=['tile', 'tile_rot_augmentation','path','tif_fn', 'patient_ID','spot_ID','label', 'spot_coord'])
    
    tiles = [] 
    if (level >= 4): #level 4 is max zoom level out for each spot in TMAs provided
        tiles = [spot.get_enclosing_box(level, return_image = False)]
        m = str(spot.TMA.name) +',' + str(spot.name) + ','
    else: 
        tiles = spot.sample_tiles(size = (300,300), level = level, overlap = .0)

    for tile in tiles:
        enlarged_tile = Spot.enlarge_tile(*tile)
        #only add tile if its generated image is valid and there is enough tissue
        img = spot.get_tile(*enlarged_tile)
        result = Spot.mask_outside_of_enclosed_cyrcle(img)
        scarce_tissue = False
        if skip_scarce_tissue:
            scarce_tissue = Spot.check_scarce_tissue(result)
            if scarce_tissue: 
                m = m + ','+ str(spot.ID) + ',scarce tissue '
                if verbose: print(m)
        if len(img.shape) != 3:
            m = m + ','+ str(spot.ID) + ',not valid image '
            if verbose: print(m)
        if len(img.shape) == 3 and scarce_tissue == False:
            s = df_labels.loc[df_labels['LIST_ID'] == str(spot.ID)]

            if (s.shape[0] != 0):
                if label == 'EDAD':
                    l = float(str(s.iloc[0][label]).replace(',','.'))
                else:
                    l = s.iloc[0][label]
                cod_dts = s.iloc[0].COD_DTS

                ds = ds.append({'tile': tile, 'tile_rot_augmentation': enlarged_tile, 'path': tma.img_path_tif, 'tif_fn': tma.name, 'patient_ID': cod_dts,'spot_ID': spot.ID,'label': l , 'spot_coord': spot.name}, ignore_index=True)          

            else:
                m = m + ','+ str(spot.ID) +',not in bbdd '
                if verbose: print(m)
            
    return ds


# In[22]:


def process_tma(tma,label = 'MMR', level=3, verbose = False, skip_scarce_tissue = True):
    m = ''
    ds = pd.DataFrame(columns=['tile', 'tile_rot_augmentation','path','tif_fn', 'patient_ID','spot_ID','label', 'spot_coord'])
    for spot in tma.spots:
        tiles = [] 
        if (level >= 4): #level 4 is max zoom level out for each spot in TMAs provided
            tiles = [spot.get_enclosing_box(level, return_image = False)]
            m = str(spot.TMA.name) +',' + str(spot.name) + ','
        else: 
            tiles = spot.sample_tiles(size = (300,300), level = level, overlap = .0)
            
        for tile in tiles:
            enlarged_tile = Spot.enlarge_tile(*tile)
            #only add tile if its generated image is valid and there is enough tissue
            img = spot.get_tile(*enlarged_tile)
            result = Spot.mask_outside_of_enclosed_cyrcle(img)
            scarce_tissue = False
            if skip_scarce_tissue:
                scarce_tissue = Spot.check_scarce_tissue(result)
                if scarce_tissue: 
                    m = m + ','+ str(spot.ID) + ',scarce tissue '
                    if verbose: print(m)
            if len(img.shape) != 3:
                m = m + ','+ str(spot.ID) + ',not valid image '
                if verbose: print(m)
            if len(img.shape) == 3 and scarce_tissue == False:
                s = df_labels.loc[df_labels['LIST_ID'] == str(spot.ID)]
                
                if (s.shape[0] != 0):
                    if label == 'EDAD':
                        l = float(str(s.iloc[0][label]).replace(',','.'))
                    else:
                        l = s.iloc[0][label]
                    cod_dts = s.iloc[0].COD_DTS
                    
                    ds = ds.append({'tile': tile, 'tile_rot_augmentation': enlarged_tile, 'path': tma.img_path_tif, 'tif_fn': tma.name, 'patient_ID': cod_dts,'spot_ID': spot.ID,'label': l , 'spot_coord': spot.name}, ignore_index=True)          
                    
                else:
                    m = m + ','+ str(spot.ID) +',not in bbdd '
                    if verbose: print(m)
            
    return ds


# ## Sanity check, print all failed spots and its reasons (set verbose=True in process_tma)

# ### Save list of all spots available with associated information including spot coordenates

# In[26]:


create_dataset_level_4_QC = False
if create_dataset_level_4_QC:
    lev = 4
    p_ds = f'dataset_level_{lev}_coordinates.csv'
    ds = pd.read_csv(p_ds)
    d = ds.merge(df_labels, how='left', left_on='patient_ID',right_on='COD_DTS', )

    d.loc[d.COD_DTS.isna()] = d[['tile','tile_rot_augmentation','path', 'tif_fn','patient_ID','label', 'spot_coord']].merge(df_labels, how='left', left_on='patient_ID',right_on='BIOPSIA', )



    #Drop duplicates after merge
    d = d.loc[d[['tile','tif_fn']].astype(str).drop_duplicates(subset = ['tile', 'tif_fn']).index]
    d.to_csv('dataset_level_4_QC.csv')
    
    #save list by patient 
    dp = d.loc[d[['patient_ID']].astype(str).drop_duplicates(subset = ['patient_ID']).index]
    print(dp.shape)
    dp.to_csv('dataset_level_4_patient_QC.csv')


# ### Save list of all spots available with associated information including spot coordenates

# In[27]:


create_dataset_level_4_QC = False
if create_dataset_level_4_QC:
    lev = 4
    p_ds = f'dataset_level_{lev}_coordinates.csv'
    ds = pd.read_csv(p_ds)
    d = ds.merge(df_labels, how='left', left_on='patient_ID',right_on='COD_DTS', )

    d.loc[d.COD_DTS.isna()] = d[['tile','tile_rot_augmentation','path', 'tif_fn','patient_ID','label', 'spot_coord']].merge(df_labels, how='left', left_on='patient_ID',right_on='BIOPSIA', )



    #Drop duplicates after merge
    d = d.loc[d[['tile','tif_fn']].astype(str).drop_duplicates(subset = ['tile', 'tif_fn']).index]
    d.to_csv('dataset_level_4_QC.csv')


# ### load dataset function

# In[28]:


def load_dataset(level = 4):
    lev = level
    try:
        p_ds = f'dataset_level_{lev}.csv'
        ds = pd.read_csv(p_ds)
        #ds = pd.concat([load_dataset(0)])
        ds['tile']=ds['tile'].apply(eval)
    except:
        print('failed')
        res = parallel(partial(process_tma,level = lev, verbose = False, label = 'MMR'),tmas)
        ds = pd.concat((*res,))
        ds.to_csv(p_ds,index=False)
    return ds


# ### Load New TMAs 

# In[38]:


#d = pd.concat([load_dataset(0), load_dataset(1), load_dataset(2), load_dataset(3), load_dataset(4)])
#d.to_csv('dataset_level_all_March21_no_overlap.csv')


# In[30]:





# In[ ]:




