#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:39:03 2020

################################################################################
Copyright (C) 2020  Thomas Meschede a.k.a. yeus

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

@author: Thomas Meschede a.k.a. yeus (yeusblender@gmail.com)

This texture synthesis algorithm takes inspiration from three papers and
combines their papers into a new algorithm:

- Image Quilting for Texture Synthesis and Transfer [Efros, Freeman]
    - taking the optimal-patch seam strategy and
- Fast Texture Synthesis using Tree-structured Vector Quantization [Wei, Levoy]
    - iterations, non-causal buildup local neighbourhood search
- Real-Time Texture Synthesis by Patch-Based Sampling [Liang et al.]
    - building a gaussian image pyramid combined with KD-Trees for
      fast searches
"""

import random
import numpy as np
import skimage
import skimage.io
import skimage.transform
import itertools
from functools import wraps
import time
#from pynndescent import NNDescent
#import gc
import math
import os, sys
import functools
sign = functools.partial(math.copysign, 1) # either of these
#import scipy
import shapely
import shapely.geometry
import logging
#import psutil #TODO: not easy in blender, because of a lacking Python.h
logger = logging.getLogger(__name__)

#ann_library = "pynndescent"
ann_library = "sklearn"
use_pynnd, use_sklearn=False,False
if ann_library=="pynndescent":
    import pynndescent as pynnd
    use_pynnd=True
elif ann_library=='sklearn':
    import sklearn
    import sklearn.neighbors
    use_sklearn=True
        
#def norm(x): return np.sqrt(x.dot(x))
def norm(x): return np.sqrt((x*x).sum(-1))
#need to be transposed for correct ultiplcation along axis 1
def normalized(x): return (x.T /norm(x)).T

def calc_angle_vec(u, v):
    """
    >>> u = vec((1.0,1.0,0.0))
    >>> v = vec((1.0,0.0,0.0))
    >>> calc_angle_vec(u,v)*rad
    45.00000000000001
    >>> u = vec((1.0,0.0,0.0))
    >>> v = vec((-1.0,0.0,0.0))
    >>> calc_angle_vec(u,v)*rad
    180.0
    >>> u = vec([-9.38963669e-01, 3.44016319e-01, 1.38777878e-17])
    >>> v = vec([-0.93896367, 0.34401632, 0.])
    >>> u @ v / (norm(v)*norm(u))
    1.0000000000000002
    >>> calc_angle_vec(u,v)*rad
    0.0
    """
    #angle = np.arctan2(norm(np.cross(u,v)), np.dot(u,v))
    res = np.sum(u*v) / (norm(u) * norm(v))
    t = np.clip(res,-1.0,1.0)
    angle = np.arccos(t)
    return angle

#consider replacing sklearn KDTree with scipy KDTree
#https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/
#from tqdm import tqdm
def tqdm(iterator, *args, **kwargs):
    return iterator

GB = 1.0/1024**3 #GB factor

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logger.info(f'func:{f.__name__} took: {te-ts:2.4f} sec')
        return result
    return wrap

@timing
def init_ann_index(data):
    """
    TODO: parameterize quality of ANN search
    """
    #metric='euclidean'
    metric='manhattan'
    if use_pynnd: 
        index = pynnd.NNDescent(data,metric,
                                n_neighbors=3,#steers the quality of the algorithm
                                n_jobs=-1 #-1: use all processors
                                )
    elif use_sklearn: 
        index = sklearn.neighbors.KDTree(data, metric=metric)#'l2'
    return index

def query_index(index, data, k):
    if use_pynnd: 
        idx, e = index.query([data], k=k)   
        return e,idx[0]
    elif sklearn: 
        e, idx = index.query([data], k=k)
        e, idx = e.flatten(), idx.flatten()
        return e,idx

def get_mem_limit(factor=0.5):
    #stats = psutil.virtual_memory()  # returns a named tuple
    #avilable_memory = getattr(stats, 'available')/1024**3 # available memory in GB
    #mem_limit = avilable_memory*factor
    #tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    if sys.platform=='linux':
        free_m = int(os.popen('free -t -m').readlines()[1].split()[-1])
    else: #TODO: include windows/mac
        free_m = 2000 #MB
    return free_m * factor / 1024 #in GB

def copy_img(target, src, pos, mask=None):
    """
    copy image src to target at pos

    careful! x & y are switched around here (normal order) in contrast to other
    functions of this library. order: pos=(x,y)
    """
    #TODO: handle border clipping problems
    # - when copying images that exten over "left" and "top" edges
    sh,sw,sch = src.shape
    th,tw,tch = target.shape

    i0x = np.clip(pos[0],0,tw)
    i0y = np.clip(pos[1],0,th)
    i1x = np.clip(pos[0]+sw,0,tw)
    i1y = np.clip(pos[1]+sh,0,th)
    
    #cut source patch to right size
    pw, ph  = max(i1x - i0x,0), max(i1y - i0y,0)

    if mask is None:
        tch = sch
        #print(pos)
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        target[i0y:i1y, i0x:i1x, 0:tch] = src[0:ph, 0:pw]
    else:
        m = mask
        target[i0y:i1y, i0x:i1x, 0:tch][m] = src[m]

    return target

def mask_blend(m, img1, img2):
    """
    this blends two images by blending them using a mask
    """
    mT = np.expand_dims(m,axis=2)
    b1 = img1 * (1 - mT)
    b2 = img2 * mT
    new_img = b1+b2
    return new_img

@timing
def transform_patch_grid_to_tex(res, res_patch, pg, example,
                                overlap,
                                use_quilting=True):
    """
    synthesize texture from pre-calculated grid

    overlap = (horizontal_ovelap, vertical_overlap)
    """
    #TODO: create a generate- "info" function so that information doesn't have
    # to be exchanged so often
    #TODO: adaptive overlap (wih larger patch sizes it might not
    #make enough sense anymore, because textures become too repetetive ...)
    #overlap = np.ceil((res_patch/6)).astype(int)
    #TODO: the overlap can actually be different from the overlap when searching
    # for the images. This might make some things more efficient
    # or we can just generate a standad overlap in all functions

    ch_num = example.shape[-1]

    #def draw_patch_grid():
    rpg = np.array(res_patch) - overlap
    if res is None: res = rpg * pg.shape[:2] + overlap
    target = np.zeros((res[0],res[1],ch_num), dtype = example.dtype)
    ta_map = np.zeros((res[0],res[1],3))
    
    target_map_patch_base = gen_coordinate_map(res_patch)
    onemask = np.ones(res_patch)
    for iy,ix in np.ndindex(pg.shape[:2]):
        #if (ix, iy) == (3,2): break
        x = ix * rpg[1]
        y = iy * rpg[0]
        y0,x0 = pa_coords = pg[iy,ix] #get coords of patch in example texture

        #TODO: searching new patches based on the already existing image also
        #      helps when having all sorts of constraints
        #TODO: maybe pass a "constraint-error-function" for the search function?
        #TODO: get rid of resolution parameters to make dynamic sized patches
        #      possible --> this doesn#t work with a "grid" function
        if all(pa_coords == (-1,-1)): # --> a part of the grid that is not assigned
            pa = np.zeros((res_patch[0],res_patch[1],4))
            pa[:,:,:]=(0, 0.31, 0.22, 1) #TODO: make different fill colors possible

        #get corresponding overlaps:
        if iy==0: ovs=(overlap[1],0,0,0) #first row
        elif ix==0: ovs=(0,overlap[0],0,0) #first column
        else: ovs=(overlap[1],overlap[0],0,0) #rest of the grid

        if (iy==0 and ix==0) or (not use_quilting): 
            pa = example[y0:y0+res_patch[0],x0:x0+res_patch[1]].copy()
            mask = onemask
        else:
            ta_patch = target[y:y+res_patch[0],x:x+res_patch[1]]
            pa, _, _, mask = optimal_patch(ta_patch, example, 
                                           res_patch, ovs, (y0,x0), (y,x))

        #skimage.io.imshow_collection([pa, ov_h[0], b_h, ov_v[0], b_v])
        copy_img(target, pa, (x,y))
        #print((ix,iy),pg[iy,ix])
        
        ta_map_patch = target_map_patch_base + (x0,y0,0)
        #TODO: find a better method how to use "partial" coordinate transfer
        #or in other words: "mixes" which appear for example at smoothed
        #out and blended optimal borders
        copy_img(ta_map, ta_map_patch, (x,y), mask=mask>0)
        #copy_img(ta_map, ta_map_patch, (x,y))


    return target, ta_map

def overlap_slices(overlap):
    """define slices for overlaps"""
    ovs = [np.s_[:,:overlap[0]],#left
           np.s_[:overlap[1],:],#top
           np.s_[:,-overlap[2]:],#right
           np.s_[-overlap[3]:,:]]#bottom
    return ovs

def create_optimal_patch(pa, ta, overlaps):
    """
    this function creates an optimal patch out of 1 tile with
    given number of border tiles where two overlapping regions
    are replaced by an optimal boundary from the other patches
    """
    mask = np.ones(pa.shape[:2])
    ovs = overlap_slices(overlaps)
    for ov, orient, sl, mirrored in zip(overlaps,["h","v","h","v"], ovs,
                              [False,False,True,True]):
        if ov>0: #if this boundary is used
            m = minimum_error_boundary_cut((pa[sl],ta[sl]), orient)
            if mirrored: m = 1-m
            mask[sl] = np.minimum(m,mask[sl])

    new_pa = mask_blend(mask, ta, pa)
    #TODO: only return mask
    return new_pa, mask

def optimal_patch(ta_patch, example, res_patch, overlap, pos_ex, pos_ta):
    """
    this creates optimal patches for reacangular patches with overlaps
    given in "overlap" with indics as follows:
          2
        -----
       1|   |3
        -----
          4

    the number of in "overlap" specifies the size of the overlap in pixels
    TODO: expand to arbitrarily sized boundaries
    TODO: make it possible to "move" the source patch a couple pixels
          to find a better fit
    """
    #TODO use np.s_ as indices
    y,x = pos_ta
    y0,x0 = pos_ex
    pa = example[y0:y0+res_patch[0],x0:x0+res_patch[1]].copy()
    optimpatch, mask = create_optimal_patch(pa, ta_patch, overlap)
    return optimpatch, pa, ta_patch, mask

def find_match(data, index, tol = 0.1, k=5):
    #get a horizonal overlap match
    e,idx = query_index(index,data,k)
    #e,idx = query_index(index,data,k)
    #TODO: keep an index with "weights" to make sure
    #      that patches are selected equally oftern OR according
    #      to a predefined distribution
    # TODO: make sure patches can get selected with an additional error-term
    # TODO: for very large textures it might make sense
    #       to only query a close radius
    # TODO: implement theshold function
    #or choose ithin a certain tolerance
    if k>1:
        min_err = e[e>0].min() #find minimum error except for the exact matching pic
        #i2 = err.argmin()
        th = min_err * (1+tol) #threshold
        idx = [i for i,e in zip(idx,e) if e<th]
        return random.choice(idx)
    else:
        return idx[0]
    #if len(idx)>1: print(f"{len(idx)},",)

@timing
def synthesize_grid(example, res_patch, res_grid, overlap, tol = 0.1, k=5):
    """
    synthesize a grid of patch indices
    by searching match pairs & triples in the
    overlaps database

    tolerance -> this basically controls the randomness of
                 the algorithm
    """

    res_patch = np.array(res_patch).astype(int)
    res_ex = example.shape[:2]
    #TODO: decrease overlap to mitigate memory_problems
    max_co = res_ex - res_patch
    ch = example.shape[-1]
    cellsize = example.itemsize
    #TODO: replace by a better KD Tree implementation or a Ball Tree
    cellsize = 8 #bytes (this is because of scikit-learns KDTree implementation
                 #       which will convert the supplied data into a float
                 #       representation)
    #ex_img_ratio = res_ex[0]/res_ex[1]

    #calculate maximum possible patch size for given number of patches and
    rp = res_patch
    pnum = max_co.prod()

    mem_limit = get_mem_limit()
    #check memory consumption for overlap database:
    #TODO: check memory consumption instructions from sklearn webpage
    #single_overlap_memory_horizontal = \
    #    overlap[0] * res_patch[1] * ch * cellsize  #in byte
    #single_overlap_memory_vertical = \
    #    overlap[1] * res_patch[0] * ch * cellsize  #in byte

    #factor 2 comes in here, because we have a 3rd database with the combined overlaps
    #totalmemoryGB = pnum * GB \
    #    * (2 * single_overlap_memory_vertical + 2 * single_overlap_memory_horizontal)
    #augmentation_multiplicator = 2 #(for mirroring, and rotating, the data gets
                                    #) multiplied
    data_memoryGB = check_memory_requirements(example,res_patch, maxmem=mem_limit,
                              disable_safety_check=True)
    logger.info(f"using approx. {3*data_memoryGB:2f} GB in RAM.")

    #TODO: build my own search algorithm which doesn't consume as much memory
    # and can find things in an image much quicker
   #TODO: the "overlaps" are the same for left & right and top & bottom
    #      patch generation can be completly left out as they can be
    #      taken directly from the texture. For the search, only the
    #      overlaps are important.
    #TODO: maybe save the coordinates of the overlaps in a second database
    #TODO: augmented patches(mirrored, )
    #TODO: check memory consumption of patch overlaps
    #TODO: if memory consumption reaches a certain thresold, lower resolution
    #      of overlap databases (this can be done in multiple ways:
    #   - only take patches from every nth pixel,
    #   - take overlaps with 1/k the resolution of the original in the database)
    #   - use smaller patch sizes
    #   - lower the reoslution of the original image but later keep the
    #     original rsolution when stitching it back together
    #   - take a random sample of patches from the original source image
    # TODO: add patch augmentation mirroring, rotaton
    #       the problem here is: when augmenting horiz. or vert.
    #       images, the "combined" would becom very large
    #       basically squared. so if we mirror vert. overlaps,
    #       only horizontally, we also have twice as many combined.
    #       If we mirror vertically & horozontally we have 3 times as
    #       many (all data).
    #       If we also mirror horiz. overlaps in th same way we have
    #       3x3 = 9x as much data in combined and 3+3=6 times as much in
    #       vert. or horiz.
    #       we can find out whether something is mirrored, rotated or
    #       whatever by analyzing the index module by the number of
    #       augentations
    #lm = []
    try:
        logger.info("init kdtree1")
        ld = create_patch_data(example, (rp[0], overlap[1]), max_co)
        l = init_ann_index(ld)
        logger.info("init kdtree2")
        td = create_patch_data(example, (overlap[0],rp[1]), max_co)
        t = init_ann_index(td)
        logger.info("init kdtree3")
        lt = init_ann_index(np.hstack((ld,td)))
        #TODO: check memory consumption of KDTrees
        #sklearn.neighbors.KDTree.valid_metrics
        #ov_db = [sklearn.neighbors.KDTree(i, metric='euclidean') for i in (ov_l, ov_t, ov_lt)]
        logger.info("KD-Tree initialization done")
    except MemoryError as err:
        logger.info(err)
        logger.info("example texture too large, algorithm needs "
           "too much RAM: {totalmemoryGB:.2f}GB")
        raise

    pg = np.full(shape=(*res_grid,2), fill_value=-1, dtype=int)
    pg[0,0] = idx2co(random.randrange(pnum), max_co) #get first patch
    #pg[0,0]=(0,0)

    #TODO: synthesize an arbitrary pic with the overlap database generated from example
    # but using a different source texture to "mix" two textures

    for i in tqdm(range(1,res_grid[1]),"find matches: first row"):
        y,x = pg[0,i-1]
        #patch = example[y:y+rp[1],x:x+rp[0]]
        #((patches[idx] - patch)**2).sum() == 0 #they have to be the same!
        ovl = example[y:y+rp[0],x:x+rp[1]][:,-overlap[0]:].flatten()
        #ov_idx = idx + (res_patch[0] - overlap[0])*max_co[1]#move towards the right by res_grid
        #(ovl-ov_l[ov_idx])
        new_idx = find_match(ovl, l , tol=tol, k=k)
        #if new_idx
        pg[0,i] = idx2co(new_idx, max_co)

    for i in tqdm(range(1,res_grid[0]),"find matches: first column"):
        y,x = pg[i-1,0]
        ovt = example[y:y+rp[1],x:x+rp[0]][-overlap[1]:,:].flatten()
        pg[i,0] = idx2co(find_match(ovt, t, tol=tol, k=k), max_co)

    for ix,iy in tqdm(itertools.product(range(1,res_grid[1]),
                                        range(1,res_grid[0])),
                      total = (res_grid[1]-1)*(res_grid[0]-1),
                      desc = "find matches: complete grid"):
        y,x = pg[iy,ix-1]
        ovl = example[y:y+rp[0],x:x+rp[1]][:,-overlap[1]:].flatten()
        y,x = pg[iy-1,ix]
        ovt = example[y:y+rp[0],x:x+rp[1]][-overlap[0]:,:].flatten()
        ovlt = np.hstack((ovl, ovt))
        pg[iy,ix] = idx2co(find_match(ovlt, lt, tol = tol, k=k), max_co)

    return pg

def minimum_error_boundary_cut(overlaps, direction):
    """
    create an optimal boundary cut from
    an error matrix calculated from overlaps
    """
    #TODO: create minimum boundary for very small overlaps (for example just 1 pixel)
    ol1, ol2 = overlaps
    #calculate error and convert to grayscale
    err = ((ol1 - ol2)**2).mean(2)

    if direction == "v": err = err.T
    minIndex = []
    E = [list(err[0])]
    for i in range(1, err.shape[0]):
        # Get min values and args, -1 = left, 0 = middle, 1 = right
        e = [np.inf] + E[-1] + [np.inf]
        e = np.array([e[:-2], e[1:-1], e[2:]])
        # Get minIndex
        minArr = e.min(0)
        minArg = e.argmin(0) - 1
        minIndex.append(minArg)
        # Set Eij = e_ij + min_
        Eij = err[i] + minArr
        E.append(list(Eij))

    # Check the last element and backtrack to find path
    path = []
    minArg = np.argmin(E[-1])
    path.append(minArg)

    # Backtrack to min path
    for idx in minIndex[::-1]:
        minArg = minArg + idx[minArg]
        path.append(minArg)

    # Reverse to find full path
    path = path[::-1]
    m = np.zeros(err.shape, dtype=float)#define mask
    for i,pi in enumerate(path):
        #p1[i,-overlap[0] + pi,0]*=0.5
        m[i,pi:]=True
        m[i,pi]=0.5 #set a "smooth" boundary
        #err[i,pi]+=0.05

    if direction=="v": return m.T
    else: return m

def create_patch_data(example, res_patch, max_co=None):
    """max_co is needed in the case where only overlap
    areas of patches are of interest. In this case we want
    the overlap areas to not extend beyond the area of
    a contiguos patch
    """
    if max_co is None: max_co = np.array(example.shape[:2]) - res_patch
    rp = res_patch
    data = np.ascontiguousarray([example[y:y+rp[0],x:x+rp[1]].flatten()
        for y,x in tqdm(np.ndindex(*max_co), "create_patch_data")])
    return data

def idx2co(idx, max_co):
    yp = int(idx/max_co[1])
    xp = idx - yp * max_co[1]
    return yp,xp

#create patches
def gen_patches(image, res_patch):
    res_img = np.array(image.shape[:2])
    max_co = res_img - res_patch
    patches = np.array([image[y:y+res_patch[0],x:x+res_patch[1]]
                        for y,x in np.ndindex(*max_co)])
    return patches

#create patches
def gen_patches_from_mask(image, mask):
    res_img = np.array(image.shape[:2])
    m_res = mask.shape[:2]
    max_co = res_img - m_res + (1,1)
    patches = np.array([image[y:y+m_res[0],x:x+m_res[1]][mask].flatten()
                        for y,x in np.ndindex(*max_co)])
    def idx2co(idx):
        yp = int(idx/max_co[1])
        xp = idx - yp * max_co[1]
        return xp,yp

    return patches, max_co, idx2co

#TODO: checkout the quality of the results for small image resolutions.
#       - it might make sense toreduce the resolution for images based
#         on certain criteria (for example patch size) anyways
def build_gaussian_pyramid(example0, min_res=8):
    #build a gaussian image pyramid
    py = [im for im in skimage.transform.pyramid_gaussian(example0, multichannel=True)]
    #filter for minimum resolution
    #min_res = 8
    #py = [im for im in py if min(im.shape[:2]) >= min_res]
    return list(reversed(py))


def create_mask_tree(img, kind="causal5x3"):
    if kind == "causal5x3": #generate causal mask
        mask_res = (3,5)
        mask = np.ones(mask_res, dtype=bool)
        mask[2,2:]=False
        mask_center = (2,2) #starting index from 0
    elif kind == "noncausal5x5": #generate non-causal mask:
        mask_res = (5,5)
        mask = np.ones(mask_res, dtype=bool)
        mask_center = (2,2) #starting index from 0
        mask[mask_center]=False
    elif kind == "noncausal3x3": #generate non-causal mask:
        mask_res = (3,3)
        mask = np.ones(mask_res, dtype=bool)
        mask_center = (1,1) #starting index from 0
    else: raise ValueError(f"kind: {kind} is unknown")

    logger.info("generating patches")
    m_patches, max_co, idx2co = gen_patches_from_mask(img,mask)
    #build local neighbourdhood KDTree for pyramid level
    logger.info("generating tree from patches")
    index = init_ann_index(m_patches)
    return index, mask, mask_center, idx2co


#generate image with noisy borders which later be cut off
def local_neighbourhood_enhance(target, ex, mask, tree,
                                mask_center, idx2co, initial=False,
                                seed = 0):
    #TODO: make tileable texture by copying the right sections from the
    # texture to the other sides

    np.random.seed(seed)

    target_res=np.array(target.shape[:2])
    m_res = np.array(mask.shape)
    image_range=((mask_center[0], mask_center[0] + target_res[0]),
                 (mask_center[1], mask_center[0] + target_res[1]))
    mc = mask_center
    if initial:
        t = np.random.random((target_res[0] + m_res[0]-1,
                               target_res[1] + m_res[1]-1,4))
        t[...,3] = 1.0
        t[image_range[0][0]: image_range[0][1],
          image_range[1][0]: image_range[1][1]] = target
    else:
        t = np.random.random((target_res[0] + m_res[0]-1,
                               target_res[1] + m_res[1]-1,4))
        #target_tmp[:target_cut[0],:target_cut[1]//2] = target[]
        t[...,3] = 1.0
        t[image_range[0][0]: image_range[0][1],
          image_range[1][0]: image_range[1][1]] = target

    #skimage.io.imshow_collection([target, t])
    #return

    if not initial: t2 = t.copy()
    else: t2 = t
    coords = itertools.product(range(image_range[0][0], image_range[0][1]),
                      range(image_range[1][0], image_range[1][1]))
    #TODO: do some caching of similar mask queries to speed up the
    # process quiet a bit
    for y,x in tqdm(coords, "iterating over image",
                    total = target_res.prod(),
                    smoothing=0.01):
        p = t[y-mc[0]:y-mc[0]+m_res[0],x-mc[1]:x-mc[1]+m_res[1]]
        pf = p[mask].flatten()
        [[err]], [[idx]] = tree.query([pf], k=1)
        xp,yp = idx2co(idx)
        pixel = ex[yp+mc[0],xp+mc[1]]
        t2[y,x] = pixel

        if False: #if debugging
            if (x,y)==(4,4):
                #py[-1][yp:yp+3,xp:xp+5]
                p2 = ex[-1][yp:yp+m_res[0],xp:xp+m_res[1]].copy()
                p2[~mask]=(0,0,0,1)
                p3 = np.zeros((*mask.shape,4))
                p3[mask] = m_patches[idx].reshape(-1,4)
                #skimage.io.imshow_collection([t])
                skimage.io.imshow_collection([p,target, p2,p3, t])
                break


    return t2[image_range[0][0]: image_range[0][1],
                  image_range[1][0]: image_range[1][1]]


def mask_synthesize_pixel(level, example, seed, pyramid, 
                          final_res,  target = None):
    img = pyramid[level]
    levelscale = img.shape[:2]/np.array(example.shape[:2])
    target_res = np.rint(final_res*levelscale).astype(int)
    if target is not None:
        target = skimage.transform.resize(target, target_res, anti_aliasing=False,order=0)
        kind = "noncausal5x5"
    else:
        target = np.zeros((*target_res,4), dtype=float)
        target[...,3] = 1.0
        kind = "causal5x3"


    tree, mask, mask_center, idx2co = create_mask_tree(img,kind)
    target = local_neighbourhood_enhance(target, img, mask,
                                         tree, mask_center, idx2co,
                                         initial=True, seed=seed)
    return target



def pixel_synthesize_texture(final_res, scale = 1/2**3, seed = 15):
    #TODO: choose output resolution level with a dedicated (2D-)scale-factor
    #       or a given resolution parameter (original texture will be
    #       up/down-scaled)
    example = skimage.transform.rescale(example0, scale, preserve_range=False,
                                        multichannel=True, anti_aliasing=True)
    py = build_gaussian_pyramid(example)
    start_level=5

    tas=[] #save images for debug information

    target = mask_synthesize_pixel(start_level, example, seed, py, final_res)
    tas.append(target)
    for level in range(start_level+1,len(py)):
        logger.info(f"\nstarting next level calculation: {level}\n")
        target = mask_synthesize_pixel(level, example, seed, py, final_res, target)
        tas.append(target)

    return target, tas



def calc_lib_scaling(res_ex, max_pixels):        
    ex_pixels = np.prod(res_ex)
    px_ratio = ex_pixels/max_pixels
    scaling = 1/math.sqrt(px_ratio)
    if scaling<1.0: return scaling
    else: return 1.0

def normalize_picture(example0, max_pixels = 256*256):
    """
    scale picture down to a maximum number of pixels
    """
    scaling = 1.0
    res_ex = example0.shape[:2]
    if max_pixels < np.prod(res_ex):
        #max_pixels basically defines whats possible with the avialable
        # memory & CPU power 256x256 has proved to be effective on modern systems
        scaling = calc_lib_scaling(res_ex, max_pixels)
        logger.info(f"resizing with scaling {scaling}")
        example = skimage.transform.rescale(example0, scaling,
        #example = skimage.transform.resize(example0, (256,256),
                                            anti_aliasing=True,
                                            multichannel=True,
                                            preserve_range=True)#.astype(np.uint8)
        #search_res = example.shape[:2]
    else: example = example0

    return example, scaling

def create_patch_params2(res_ex, scaling,
                         overlap_ratio, patch_ratio):
    """patch_ratio = #size of patches in comparison with original
    """
    #TODO: define minimum size for patches
    res_patch0 = int(min(res_ex)*patch_ratio)
    res_patch0 = np.array([res_patch0]*2)
    res_patch = np.round(res_patch0*scaling).astype(int)
    overlap = np.ceil(res_patch*overlap_ratio).astype(int)
    #res_patch2 = np.round(np.array(res_patch)/scaling).astype(int)
    overlap0 = np.ceil(res_patch0*overlap_ratio).astype(int)

    return res_patch, res_patch0, overlap, overlap0

def create_patch_params(example0, scaling,
                        overlap_ratio = 1/6, patch_ratio = 0.05):
    return create_patch_params2(example0.shape[:2], scaling,
                         overlap_ratio, patch_ratio)

#create test-target to fill with mask:
def generate_test_target_with_fill_mask(example):
    target = example.copy()
    target[...,3]=1.0

    verts = [np.array(((0.1,0.1),(0.4,0.15),(0.41,0.4),(0.2,0.38))),
            np.array(((0.5,0.55),(0.85,0.53),(0.8,0.7),(0.51,0.71)))]
    masks = []
    pxverts = []
    for v in verts:
        pxverts.append(v * target.shape[:2])
        rr,cc = skimage.draw.polygon(*v.T)
        mask = np.zeros(target.shape[:2])
        mask[rr,cc]=1.0
        target[rr,cc]=(1,0,0,1)
        masks.append(mask)

    return target, masks, pxverts


def draw_polygon_mask(verts, size):
    rr,cc = skimage.draw.polygon(*verts.T)
    mask = np.zeros(size)
    mask[rr,cc]=1.0
    return mask

def edge_distance(poly, x,y):
    d = poly.boundary.distance(shapely.geometry.Point(x,y))
    if poly.contains(shapely.geometry.Point(x,y)): return d
    else: return -d


def get_poly_levelset(verts, width=10):
    poly = shapely.geometry.Polygon(verts)
    poly_box = poly.buffer(+width) #add two pixels on the container
    bbox = poly_box.bounds
    miny, minx, maxy, maxx = bbox_px = np.round(np.array(bbox)).astype(int)
    w,h = maxx - minx, maxy-miny


    bbcoords = itertools.product(range(miny,maxy), range(minx, maxx))
    levelset = np.array([edge_distance(poly,y,x) for y,x in bbcoords]).reshape(h,w)
    #normalize levelset:
    #levelset = np.maximum(levelset/levelset.max(),0.0)
    return levelset, bbox_px

@timing
def fill_area_with_texture(target, example0, ta_map_final=None,
                           patch_ratio=0.1, libsize = 128*128,
                           verts=None, mask = None, bounding_box = None):
    if bounding_box is None:
        area = shapely.geometry.Polygon(verts)
        ov = 1 #overlap
        y0,x0,y1,x1 = np.array(area.bounds).astype(int) + (-ov,-ov,ov,ov)
    else:
        y0,x0,y1,x1 = bounding_box
    #print("create levelset")
    #levelset, (minx, miny, maxx, maxy) = get_poly_levelset(verts, width=ov)
    bbox = target[y0:y1,x0:x1]
    #bmask = levelset>0
    if mask is None:
        mask = draw_polygon_mask(verts,target.shape[:2])
        bmask = mask[y0:y1,x0:x1]>0
    else:
        bmask = mask
    #bmask2 = mask[y0:y1,x0:x1]>0
    #area.boundary.buffer(100)

    logger.info("synthesize texture")
    fill1, ta_map = synth_patch_tex(bbox, example0, k=1,
                                          patch_ratio=patch_ratio, 
                                          libsize=libsize)
    copy_img(target, fill1, (x0,y0), bmask)
    #ta_map_final = np.full((*target.shape[:2],3),[0,0,0])
    #ta_map_final = np.full([*target.shape[:2],3],0)
    if ta_map_final is None: ta_map_final = np.zeros([*target.shape[:2],3])
    #copy_img(ta_map_final, ta_map, (x0,y0), bmask)
    copy_img(ta_map_final, ta_map, (x0,y0), bmask)
    #TODO: somehow the copy operation doesnt work here
    #import ipdb; ipdb.set_trace() # BREAKPOINT

    return target, ta_map_final

def calculate_memory_consumption(res_ex, res_patch, 
                                 ch_num, itemsize):
    patch_num=np.product(np.array(res_ex) - res_patch)
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    data_memoryGB = patch_num*ch_num*itemsize*np.product(res_patch)*GB
    #print(f"using approx. {data_memoryGB:2f} GB in RAM.")
    return data_memoryGB

def check_memory_requirements(example, res_patch, maxmem = 1.0,
                              disable_safety_check=False):
    data_memoryGB = calculate_memory_consumption(
            np.array(example.shape[:2]),res_patch,
            ch_num = example.shape[2], 
            itemsize = example.itemsize)
    logger.info(f"using {data_memoryGB:.4f} of {maxmem} GB for synthesis")
    if not disable_safety_check:
        if data_memoryGB > maxmem:
            raise MemoryError("the algorithm would exceed the "
                          "maximum amount of Memory: " 
                          f"{data_memoryGB:2f} GB,; max: {maxmem}")

    return data_memoryGB

@timing
def prepare_tree(example0, lib_size, overlap_ratio, patch_ratio, 
                 mode=None):
    example, scaling = normalize_picture(example0, lib_size)
    res_patch, res_patch0, overlap, overlap0 = create_patch_params(example0, scaling,
                                                                   overlap_ratio,
                                                                   patch_ratio)
    max_co = np.array(example.shape[:2]) - res_patch

    data_memoryGB = check_memory_requirements(example,res_patch, 
                                              maxmem=get_mem_limit(),
                                              disable_safety_check=True)
    logger.info(f"using approx. {data_memoryGB:2f} GB in RAM.")
    try:
        if mode==None:
            data = create_patch_data(example, res_patch, max_co)
            index = init_ann_index(data)
        else:
            if ('both' in mode) or ('horizontal' in mode):
                logger.info("init kdtree1")
                ld = create_patch_data(example, (res_patch[0], overlap[1]), max_co)
                l = init_ann_index(ld)
            if ('both' in mode) or ('vertical' in mode):
                logger.info("init kdtree2")
                td = create_patch_data(example, (overlap[0],res_patch[1]), max_co)
                t = init_ann_index(td)
            if 'both:':
                logger.info("init kdtree3")
                lt = init_ann_index(np.hstack((ld,td)))
            index = [l,t,lt]
    except MemoryError as err:
        logger.info(err)
        logger.info("example texture too large, algorithm needs "
           "too much RAM: {totalmemoryGB:.2f}GB")
        raise    
    return index, res_patch, res_patch0, overlap, overlap0, max_co, scaling

def gen_coordinate_grid(shape, flatten=False):
    x = np.arange(0,shape[1])
    y = np.arange(0,shape[0])
    #get coordinate grid
    #TODO: this might be more elegant using np.dstack
    grid = np.stack(np.meshgrid(y,x),axis = 2)
    if flatten: grid = grid.reshape(-1,2)
    return grid

def gen_coordinate_map(shape):
    grid2ch=gen_coordinate_grid(shape)
    #TODO: this might be more elegant using np.dstack
    grid3ch=np.stack((grid2ch[...,0],
                      grid2ch[...,1],
                      np.zeros(shape)),axis=2)#,axis=2)
    return grid3ch

"""
#TODO: high quality optimization
This function doesn't work yet...

the main problem is that when changing the resolution of textures
only for single patches, we get a different result for that area
then if we would change it for the entire texture at once. The reason
for this are the rounding errors when going to smaller resolution
pictures: some of the pixels of the smaller sized image will include
the values of multiple pixels also from the neighbouring patches. if
we decrease the resolution of only a patch, this does not happen.

This
makes it impossible to incoporate 
@timing
def synth_patch_tex(target0, example0, 
                    lib_size = 10000,
                    k=1, 
                    patch_ratio=0.1,
                    overlap_ratio = 1/6,
                    tol=0.1
                    ):
    '''
    lib_size = 10000
    k=1
    patch_ratio=0.1
    overlap_ratio = 1/6
    tol=0.1
    '''
    #TODO: merge this function with the "search" and the "optimal patch"
    # functionality to make it similar to the "synthesize_tex_patches" function
    chan = 3
    #target = target.copy()
    example0 = example0[...,:chan]
    target_map = target0.copy()
    (trees, res_patch,
     res_patch0, overlap,
     overlap0, max_co,
     scaling) = prepare_tree(example0, lib_size, overlap_ratio, patch_ratio,
                            mode=["horizontal","vertical","both"])
    logger.info(f"patch_size: {res_patch0}; initial scaling: {scaling}, ")

    #define search grid
    res_target0 = target0.shape[:2]
    rpg0 = np.array(res_patch0) - overlap0
    rpg = np.array(res_patch) - overlap
    res_grid0 = np.ceil((res_target0-overlap0)/rpg0).astype(int)

    co_map_base = gen_coordinate_map(res_patch0)

    #resize target to the same scale as the scaled example
    #this is actually important and can NOT be done "on the fly" for
    #each 
    target = skimage.transform.rescale(target0, scaling,
                                        anti_aliasing=True,
                                        multichannel=True,
                                        preserve_range=True)#.astype(np.uint8)

    example = skimage.transform.rescale(example0, scaling,
                                        anti_aliasing=True,
                                        multichannel=True,
                                        preserve_range=True)#.astype(np.uint8)

    #left corner (0,0)
    for coords in tqdm(np.ndindex(*res_grid0),"iterate over image"):
        yp,xp=np.array(coords) * rpg
        search_area0 = target[yp:yp+res_patch[0],
                              xp:xp+res_patch[1]].copy()
        if coords==(0,0):
            pa_coords_idx = pa_y,pa_x = np.array(idx2co(
                                    random.randrange(np.product(max_co)), 
                                    max_co)) #get first patch
           
            #skimage.io.imshow_collection([patch, co_map/(*example.shape[:2],1)])
        elif coords[0]==0: #first row
            ovl = search_area0[:,:overlap[1]]
            #ovl = skimage.transform.resize(ovl,
            #                               (res_patch[0],overlap[1]),
            #                               preserve_range=True)
            pa_idx = find_match(ovl.flatten(), trees[0] , tol=tol, k=k)
            pa_coords_idx = pa_y,pa_x = np.array(idx2co(pa_idx, max_co))

        pa = example[pa_y:pa_y+res_patch[0],pa_x:pa_x+res_patch[1]]
        copy_img(target,pa,(xp,yp))
        pa_y0,pa_x0 = np.round(pa_coords_idx / scaling).astype(int)
        pa0 = example0[pa_y0:pa_y0+res_patch0[0],pa_x0:pa_x0+res_patch0[1]]
        copy_img(target0,pa0,(xp,yp))
        co_map = co_map_base + (pa_y0,pa_x0,0)
        copy_img(target_map,co_map,(xp,yp))
            
        #ovl = example[y:y+rp[0],x:x+rp[1]][:,-overlap[0]:].flatten()
        
        #if False:
        if coords==(0,10):
            #patch_idx = trees[0].get_arrays()[0][pa_idx].reshape(res_patch[0],-1,3)
            patch_idx = trees[0]._raw_data[pa_idx].reshape(res_patch[0],-1,3)
            skimage.io.imshow_collection([ovl,search_area0, pa0, patch_idx])
            skimage.io.imshow_collection([pa0, target_map/(*example0.shape[:2],1), target])
            break

    return target_map, target
"""


def synth_patch_tex(target, example0, k=1, patch_ratio=0.1, libsize = 256*256):
    #TODO: merge this function with the "search" and the "optimal patch"
    # functionality to make it similar to the "synthesize_tex_patches" function
    example, scaling = normalize_picture(example0, libsize)
    res_target = target.shape[:2]
    res_patch, res_patch2, overlap, overlap2 = create_patch_params(example0, scaling,
                                                        patch_ratio=patch_ratio)
    res_grid = np.ceil(res_target/(res_patch2 - overlap2)).astype(int)

    logger.info(f"patch_size: {res_patch2}; initial scaling: {scaling}, ")

    #time.sleep(10.0)

    pg = synthesize_grid(example, res_patch, res_grid, overlap, tol=.1, k=k)

    #transform pg coordinates into original source texture
    pg2 = np.round(pg / scaling).astype(int)
    pgimage = (pg2/example0.shape[:2])[...,::-1]
    pgimage = np.dstack((pgimage,np.zeros(pgimage.shape[:2])))

    target, ta_map = transform_patch_grid_to_tex(None, res_patch2, pg2, example0,
                                         overlap2,
                                         use_quilting=True)

    #target2, ta_map = transform_patch_grid_to_tex(None, res_patch, pg, example,
    #                                      overlap,
    #                                      use_quilting=True)

    return (target[:res_target[0],:res_target[1]], 
            ta_map[:res_target[0],:res_target[1]])



@timing
def synthesize_tex_patches(target0, example0,
                           lib_size = 200*200,
                            patch_ratio = 0.07,
                            overlap_ratio = 1/3):
    """
    TODO: make target texture use the already synthesized patch as an option
    --> this means to take a patch from the high-res image and scale it down
    "on the fly". This way can avoid creating a scaled down version of the
    target. It also makes the produced patch more "precise" as we are not
    restricted to the target-pixels anymore
    """
    
    target_new = target0.copy()
    (tree, res_patch,
     res_patch0, overlap,
     overlap0, max_co, 
     scaling) = prepare_tree(example0, lib_size, overlap_ratio, patch_ratio)

    res_target0 = target0.shape[:2]
    rpg0 = np.array(res_patch0) - overlap0
    res_grid0 = np.ceil((res_target0-overlap0)/rpg0).astype(int)

    rp = res_patch
    #resize target to the same scale as the scaled example
    #TODO: replace this with a "instant-scale-down" method (have a look
    #at function documentation)
    target = skimage.transform.rescale(target0, scaling,
                                        anti_aliasing=True,
                                        multichannel=True,
                                        preserve_range=True)#.astype(np.uint8)

    for coords in tqdm(np.ndindex(*res_grid0),"iterate over image"):
        #TODO: only iterate over "necessary" pixels indicated by a mask
        #to get "whole" patches, we need the last row to have the same
        #border as the target image thats why we use "minimum":
        #print(coords)
        yp0,xp0 = co_target0 = np.minimum(res_target0-res_patch0,np.array(coords) * rpg0)
        yp,xp  = np.round(co_target0*scaling).astype(int)
        yp,xp = np.minimum(np.array(target.shape[:2])-res_patch, (yp,xp))

        #TODO: replace with "real" target and scale down. This makes
        # the search more precise as pixel rounding effects can be
        # mitigated this way
        search_area = target[yp:yp+rp[0],xp:xp+rp[1]].copy()
        new_idx = find_match(search_area.flatten(), tree , tol=0.1, k=1)
        #find patch from original image
        co_p = np.array(idx2co(new_idx, max_co))
        co_p0 = np.round(co_p / scaling).astype(int)
        ovs = np.r_[overlap0,0,0]#,overlap0]
        pa, pa0, ta0, mask = optimal_patch(target_new, example0, res_patch0,
                                     ovs, co_p0, co_target0)
        copy_img(target_new, pa, co_target0[::-1])

    return target_new

def calculate_face_normal(f):
    v1,v2 = f[1]-f[0], f[2]-f[1]
    n = sign(np.cross(v1,v2))
    return n

#@timing
def transfer_patch_pixelwise(target, search_area0, 
                             yp,xp,
                             edge_info,
                             fromtarget,
                             face_source,
                             ta_map = None,
                             sub_pixels = 1,
                             mask = None,
                             pa = None,
                             ta_pa = None,
                             tol=0.0):
    """This function takes an edge and pixel, and searches for the corresponding
    pixel at another edge.
    
    TODO: cythonize or do other stuff with this function (nuitka, scipy, numba)
    """
    e1,e2,v1,v2,e2_perp_left = edge_info
    """
    TODO: the below version is still present, as it might
    be preferable to do a loop-based version using cython
    for patch_index in np.ndindex(search_area0.shape[:2]): 
      for sub_pix in np.ndindex(sub_pixels,sub_pixels):
        sub_idx =  np.array(patch_index) + np.array(sub_pix)/sub_pixels
        coords = np.array((yp,xp)) + sub_idx - e1[0]
        c = coords.dot(v1)/(v1.dot(v1))#orthogonal projection of pixel on edge
        d = c*v1-coords #orthogonal distance from edge
        #pixel on left or right side of edge?
        isleft = (v1[1]*d[0]-d[1]*v1[0])>0
        d_len = norm(d)
        if isleft or (d_len<tol):
            #calculate corresponding pixel at other edge in reverse direction
            # (this is why we use (1-c))
            p_e2 = (1-c)*(v2) + e2[0]
            px2_coords = p_e2 + d_len * e2_perp_left * (1 if isleft else -1)
            tmp = np.round(px2_coords).astype(int)
            if fromtarget and isleft:
                #copy pixel from target to the search index
                #TODO: use interpolation here (if edge lengths differ in length)
                search_area0[patch_index] = target[tuple(tmp)]
            elif check_inside_face(face_source, tmp, tol=tol):
                if mask[patch_index]>0:
                     #copy pixel from generated patch back to target
                    target[tuple(tmp)] = pa[patch_index]"""

    y1,x1 = search_area0.shape[:2]
    x = np.arange(0,x1,1.0/sub_pixels)
    y = np.arange(0,y1,1.0/sub_pixels)
    #get coordinate grid
    sub_pix = np.stack(np.meshgrid(y,x),axis = 2).reshape(-1,2)
    #vector from uv edge to sub_pixel in patch with corner yp,xp
    coords_matrix = (yp,xp) + sub_pix - e1[0] 

    
    proj = proj = coords_matrix.dot(v1)/(v1.dot(v1)) #orthogonal projection of pixels on edge
    orth_d = np.outer(proj,v1)-coords_matrix #orthogonal distance vector from edge
    
    isleft = (v1[1]*orth_d[:,0]-orth_d[:,1]*v1[0])>0
    d_len = norm(orth_d)#distance of pixel from edge

    #calculate corresponding pixel at other edge in reverse direction
    # (this is why we use (1-proj))    
    p_e2s = np.outer((1-proj),(v2)) + e2[0] #projecton on corresponding edge
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    px2_cos = p_e2s + (np.outer(d_len,e2_perp_left).T*(isleft * 2 - 1)).T#orthogonal pixel from that projection point on the edge
    px2_cos = np.round(px2_cos).astype(int)
    
    #choose to transfer pixels from the left soude + a little overlap
    #transfer_pixels = (isleft | (d_len<tol)).reshape(y1,-1)
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    #coords = np.array((yp,xp)) + sub_idx - e1[0]

    #TODO: either cythonize or vectorize using numpy
    #TODO: put in some checks whether px2coords are "valid"
    # we can do this by checking all coordinates for bounds
    # create a mask from this and apply it to all the
    # vectors that are being zipped in the below loop
    if fromtarget:
        for sp, isleft, px2_coords, d_len in zip(sub_pix, isleft, px2_cos, d_len):
            #import ipdb; ipdb.set_trace() # BREAKPOINT
            if isleft:
                patch_index = tuple(sp.astype(int))
                search_area0[patch_index] = target[tuple(px2_coords)]
        #left_mask = isleft.reshape(y1,-1)
        #search_area0[left_mask] = target[px2_cos[:,1],px2_cos[:,0]][isleft]
    else:
        for sp, isleft, px2_coords, d_len in zip(sub_pix, isleft, px2_cos, d_len):
            #import ipdb; ipdb.set_trace() # BREAKPOINT
            if isleft or (d_len<tol):
              if check_inside_convex_quadrilateral(face_source, px2_coords, tol=tol):
                patch_index = tuple(sp.astype(int))
                if mask[patch_index]>0:
                   #copy pixel from generated patch back to target
                   #target[tuple(px2_coords)][:3] = pa[patch_index][:3]*mask[patch_index]
                   target[tuple(px2_coords)] = pa[patch_index]
                   if ta_map is not None:
                       ta_map[tuple(px2_coords)] = ta_pa[patch_index]

def check_inside_convex_quadrilateral(corners, p, tol=0):
    """be careful!, because this function is based
    on whether the coordinate system is left- or right-handed"""
    #TODO: vectorize this function or implement some cython-magic
    sv = np.roll(corners, 1, 0) - corners
    pv = p-corners

    #normalized normal vetors on all edges:
    n = normalized(sv[:,::-1]*(1,-1))
    #orthogonal distance for all edges
    orth_d = (pv*n).sum(axis=1)
    #calculate distance from all edges
    
    return np.all(orth_d < tol)
#    else:
#        #check if point lies on left side of every side vector
    #return np.all(sv[:,0] * pv[:,1] - sv[:,1] * pv[:,0] >= 0) # > 0    
    #return True

#cache this function as it will be very similar for many points in the
#polygon. The function cache should be reset when the algorithm gets rerun though
def check_inside_face(polygon, point, tol=0.0):
        face = shapely.geometry.Polygon(polygon).buffer(tol)
        return face.contains(shapely.geometry.Point(*point))

@timing
def make_seamless_edge(edge1,edge2, target, example0, ta_map, 
                       patch_ratio, 
                       lib_size, debug_level=0,
                       tree_info = None):
    #TODO: make sure that the "longer" edge is defined as
    #"e1" --> this is so that we don't have huge undefined patch_sizes
    #in the final result. Additionally, this make sure that we always have a
    #"high-resolution"-edge where the patchesa re used.
    
    
    (e1,verts1),(e2,verts2) = edge1,edge2
    v1 = e1[1]-e1[0]
    v2 = e2[1]-e2[0]
    if norm(v1)<norm(v2): #if edge1 is shorter than edge2 (in pixels)
        (e1,verts1),(e2,verts2) = edge2,edge1
        v1 = e1[1]-e1[0]
        v2 = e2[1]-e2[0]
    tol=2.0

    #edge_perpendivular vectors pointing to the "left" of an edge
    e2_perp_left = normalized(v2[::-1]*(1,-1))

    #move along the edge and generate patches from information
    #from both sides of the edge
    #TODO: make overlap ratio parameterizable
    overlap_ratio = 1/6
    
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    if tree_info is None:
        (tree, res_patch,
         res_patch0, overlap,
         overlap0, max_co, 
         scaling) = tree_info = prepare_tree(example0, lib_size, overlap_ratio, patch_ratio)
    else:
        (tree, res_patch,
         res_patch0, overlap,
         overlap0, max_co, 
         scaling) = tree_info
    logger.info(f"patch_size = {res_patch0}, overlap = {overlap0}, libsize = {lib_size}")
#    target_new = target.copy()
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    #pad the target to make sure we can do all operations at its
    #borders
    target_new = np.pad(target,((res_patch0[0],res_patch0[0]),
                                    (res_patch0[1],res_patch0[1]),(0,0)), mode='edge')
    ta_map = np.pad(ta_map,((res_patch0[0],res_patch0[0]),
                                    (res_patch0[1],res_patch0[1]),(0,0)), mode='edge')
    #transform coordinates to padded target
    e1+=res_patch0
    e2+=res_patch0
    verts1+=res_patch0
    verts2+=res_patch0

    edge_dir = normalized(v1)
    step_width = min(res_patch0)*0.5
    target_map_patch_base = gen_coordinate_map(res_patch0)
    for counter, i in enumerate(np.arange(0,norm(v1)+step_width,step_width)):
        # TODO: for the corners we need to search the faces and edges
        # connected to this corner to fill the patc with corresponding
        # pixels
        #print(i)
        cy,cx = e1[0] + i*edge_dir #calculate center of patch
        yp,xp = np.round((cy,cx) - res_patch0/2).astype(int) #calculate left upper corner of patch
        #target_new[skimage.draw.circle(cy,cx,2)]=(1,0,0,1)
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        #TODO: make it possible to create a search area at the "border" of an
        #image right now it is necessary to create a padding for th
        #input image
        #search_area0 = np.random.random((*res_patch0,4))
        #copy_img(search_area0,
        #         target_new,#[yp:yp+res_patch0[0],xp:xp+res_patch0[1]],
        #         -pos[::-1])
        #TODO: fill search area "empty spaces" (outside uv island) with better pixels
        search_area0 = target_new[yp:yp+res_patch0[0],
                                  xp:xp+res_patch0[1]].copy()
        transfer_patch_pixelwise(target_new, search_area0, 
                                     yp,xp,
                                     edge_info = (e1,e2,v1,v2,e2_perp_left),
                                     fromtarget = True,
                                     face_source = verts2,
                                     sub_pixels = 1)

        search_area = skimage.transform.resize(search_area0,res_patch,
                                                preserve_range=True)

        new_idx = find_match(search_area.flatten(), tree , tol=0.1, k=1)
        co_p = np.array(idx2co(new_idx, max_co))
        #TODO: make a small local high-resolution search to find better matching
        # patches
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        co_p0 = np.round(co_p / scaling).astype(int)
        ovs = np.r_[overlap0,overlap0]
        pa, pa0, ta0, mask = optimal_patch(search_area0, example0, res_patch0,
                                         ovs, co_p0, (yp,xp))
        
        ta_pa = target_map_patch_base + [*co_p0[::-1],0]
        
        if debug_level>0: #for debugging
            search_area0[:,:,0:2]=(1,1)
            pa0_r = pa0.copy()
            pa0[:,:,0]*=0.5 #green -> "inside"
            pa[:,:,1]*=0.5 #red -> "outside"
        
        #copy one side of the patch back to its respective face 
        #and also create a left/rght mask for masking the second part
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        #print("copy back!")
        transfer_patch_pixelwise(target_new, search_area0, 
                                     yp,xp,
                                     edge_info = (e1,e2,v1,v2,e2_perp_left),
                                     fromtarget = False,
                                     face_source = verts2,
                                     sub_pixels = 2,
                                     mask = mask,
                                     pa = pa,
                                     ta_pa = ta_pa,
                                     ta_map = ta_map,
                                     tol=tol)
        
        
        #TODO: copy only the part thats "inside" face 1
        mask_inside = np.zeros(search_area0.shape[:2])
        for patch_index in np.ndindex(search_area0.shape[:2]):
            coords = patch_index + np.array((yp,xp))
            mask_inside[patch_index] = check_inside_convex_quadrilateral(verts1,coords, tol=tol)
            
        #copy only the right side to its place
        #mask_right_optimal = mask_sides==0
        #mask_right_optimal = np.minimum(mask_sides==0, mask>0)
        mask_right_optimal = np.minimum(mask>0, mask_inside>0)
        copy_img(target_new, pa0, (xp,yp), mask_right_optimal)
        
        copy_img(ta_map, ta_pa, (xp,yp), mask_right_optimal)

        #if counter == 2:
        if False:#debug_level>0:#counter == 2:
            #patch_from_data = data[new_idx].reshape(*res_patch,4)
            skimage.io.imshow_collection([search_area0, pa, mask, pa0, mask_inside])
            #skimage.io.imshow_collection([search_area0, search_area, target1,
            #                      patch_from_data, pa,pa0,ta0])
            #break

    #import ipdb; ipdb.set_trace() # BREAKPOINT
    return (target_new[res_patch0[0]:-res_patch0[0],
                      res_patch0[1]:-res_patch0[1]], 
            ta_map[res_patch0[0]:-res_patch0[0],
                  res_patch0[1]:-res_patch0[1]], 
            tree_info)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    logging.getLogger('tex_synthesize').setLevel(logging.INFO)
    #example0 = example = skimage.io.imread("textures/3.gif") #load example texture
    example0 = skimage.io.imread("textures/rpitex.png")
    example0 = example0/255
    #example0 = skimage.transform.resize(example0, (500,1000))
    example0 = skimage.transform.rescale(example0, 0.3, multichannel=True)
    #example0 = example = skimage.io.imread("RASP_03_05.png") #load example texture
    #TODO: more sophisticated memory reduction techniques (such as
    # a custom KDTree) This KDTree could be based on different, hierarchical
    # image resolutions for the error-search

    #TODO: test "overhaul" of images with correspondance maps
    

    #test image synthesis function
    if True:
        seed = 32
        np.random.seed(seed)
        random.seed(seed)#2992)#25 is chip + original img
        
        channels = 3 #TODO: prepare function for 1,3 and 4 channels
        target0 = np.full((800,1000,channels),0.0)
        target, ta_map = synth_patch_tex(target0,example0,
                                                libsize=256*256,
                                                patch_ratio=0.1)
        ta_map = ta_map/np.array([*example0.shape[:2][::-1],1])
        skimage.io.imshow_collection([target, ta_map])

    if False:
        np.random.seed(10)
        random.seed(50)#2992)#25 is chip + original img
        target1, mask, verts = generate_test_target_with_fill_mask(example0)
        skimage.io.imshow_collection([target1,mask])
        target1[:]=(0,0.5,0,1)
        for v in verts:
            y0,x0,y1,x1 = np.array(shapely.geometry.Polygon(v).bounds).astype(int)
            #target1[y0:y1,x0:x1]*=(0.5,0.5,0.5,1)#mark bounding box for debugging
            target1, fill1, fill2, pgimg, bmask = fill_area_with_texture(target1, example0, v)
    
        check_inside_face(verts[0],(55,100))
        
    
        #select two corresponding edges:
        edges = ((verts[0][:2],verts[1][:2]),
                 (verts[0][1:3],verts[1][1:3]))
                    
        target_new = target1
        for e1,e2 in edges[:1]:
            logger.info("alter next edge")
            target_new = make_seamless_edge((e1,verts[0]),(e2,verts[1]), 
                                            target_new, example0)
        
        skimage.io.imshow_collection([target_new, target1])
    
    #test image copying with "wrong" boundaries
    if False:
        tmp = np.full((100,100,4),(1.0,0,0,1))
        search_area0 = np.random.random((28,28,4))
        pos = (5,10)
        copy_img(target=search_area0,src=tmp,pos=pos)     
        skimage.io.imshow_collection([search_area0,tmp])
    
    #skimage.io.imshow_collection([target_new])
    #edge_area = np.array(edgebox.boundary.coords)[:-1]
    #target1[skimage.draw.polygon(*x.boundary.xy)] *= (0.5,0.5,1.0,1.0)
    #skimage.io.imshow_collection([target1])

    #edgebox.buffer(15).bounds


    #area around edge to get edge straight
    #calculate area around edge:


    #edge = 10
    #pos = y0-edge,x0-edge
    #target0 = target1[y0-edge:y1+edge,x0-edge:x1+edge].copy()

    #skimage.io.imshow_collection([target0])
    #TODO: cut out the stripes between the two uvs



    #skimage.io.imshow_collection([target0_start, fill1, fill2, pgimg])#,target0])

    #targets=[]
    #targets.append(synthesize_tex_patches(target0, example))
    #targets.append(synthesize_tex_patches(targets[-1], example,patch_ratio = 0.05))
    #targets.append(synthesize_tex_patches(targets[-1], example,patch_ratio = 0.04))

    #copy_img(target1, targets[-1], pos[::-1])

    #skimage.io.imshow_collection(targets)
    #skimage.io.imshow_collection([target0, target])
    #skimage.io.imshow_collection([pgimg])
    #skimage.io.imshow_collection([target1])
    #skimage.io.imshow_collection([example])

    #tree, mask, mask_center, idx2co = create_mask_tree(example0,"noncausal5x5")

    #skimage.io.imshow_collection([target, target2, pgimg, example0, target1, mask])
    #skimage.io.imshow_collection([target1, fill1, fill2, pgimg])
    #skimage.io.imshow_collection([target1])
    #skimage.io.imshow_collection([example0])
    #skimage.io.imsave("debug/synth.jpg", target0[...,:3])

    #analyze resulting patchgrid (og):
    #import pandas as pd
    #allcells = pg.view([('f0', pg.dtype), ('f1', pg.dtype)]).squeeze(axis=-1)
    #allcells = pd.DataFrame(np.unique(allcells.flatten(), return_counts=True)).T
    #allcells.sort_values(1)
    #pd.DataFrame.from_items(allcells)
    
