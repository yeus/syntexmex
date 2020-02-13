#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:39:03 2020

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

and some more inspiration
- wfc synthesis algorithm

optionally in the future:
- Graphcut Textures: Image and Video Synthesis Using Graph Cuts [Kwatra, Schödl]
-

"""

import random
import numpy as np
import skimage
import skimage.io
import skimage.transform
import itertools
from functools import wraps
import time
import sklearn
import sklearn.neighbors
#import gc
import math
import functools
sign = functools.partial(math.copysign, 1) # either of these
#import scipy
import shapely
import shapely.geometry

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
        print(f'func:{f.__name__} took: {te-ts:2.4f} sec')
        return result
    return wrap

def copy_img(target, src, pos, mask=None):
    """
    copy image src to target at pos

    careful! x & y are switch around here i contrast to other
    functions of this library. oder: pos=(x,y)
    """
    #TODO: handle border clipping problems
    # - when copying images that exten over "left" and "top" edges
    sh,sw,sch = src.shape
    th,tw,tch = target.shape

    i0x = pos[0]
    i0y = pos[1]
    i1x = i0x+sw
    i1y = i0y+sh
    t_ix0 = max(i0x, 0)
    t_iy0 = max(i0y, 0)
    t_ix1 = min(i1x, tw)
    t_iy1 = min(i1y, th)

    #cut patch to right size
    pw, ph  = t_ix1 - t_ix0, t_iy1 - t_iy0

    if mask is None:
        tch = sch
        #print(pos)
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        target[t_iy0:t_iy1, t_ix0:t_ix1, 0:tch] = src[0:ph, 0:pw]
    else:
        m = mask
        target[t_iy0:t_iy1, t_ix0:t_ix1, 0:tch][m] = src[m]

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

        if (iy==0 and ix==0) or (not use_quilting): pa = example[y0:y0+res_patch[0],x0:x0+res_patch[1]].copy()
        else:
            ta_patch = target[y:y+res_patch[0],x:x+res_patch[1]]
            pa,_,_,_ = optimal_patch(ta_patch, example, res_patch, ovs, (y0,x0), (y,x))

        #skimage.io.imshow_collection([pa, ov_h[0], b_h, ov_v[0], b_v])
        copy_img(target, pa, (x,y))
        #print((ix,iy),pg[iy,ix])

    return target

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

def find_match(data, db, tol = 0.1, k=5):
    #get a horizonal overlap match
    e, idx = db.query([data], k=k)
    e, idx = e.flatten(), idx.flatten()
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
    mem_limit = 3.0 #GB

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
    check_memory_requirements(example,res_patch, maxmem=1.0)

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
        print("init kdtree1")
        l = create_patch_data(example, (rp[0], overlap[1]), max_co)
        l = sklearn.neighbors.KDTree(l, metric='l2')
        print("init kdtree2")
        t = create_patch_data(example, (overlap[0],rp[1]), max_co)
        t = sklearn.neighbors.KDTree(t, metric='l2')
        print("init kdtree3")
        lt = sklearn.neighbors.KDTree(np.hstack((l.get_arrays()[0],
                                                t.get_arrays()[0])),
                                      metric='l2')
        #TODO: check memory consumption of KDTrees
        #sklearn.neighbors.KDTree.valid_metrics
        #ov_db = [sklearn.neighbors.KDTree(i, metric='euclidean') for i in (ov_l, ov_t, ov_lt)]
        print("KD-Tree initialization done")
    except MemoryError as err:
        print(err)
        print("example texture too large, algorithm needs "
           "too much RAM: {totalmemoryGB:.2f}GB")
        raise

    #TODO: delete ov_l, ov_t, ov_lt to free up memory (maybe just put them
    # into a function so that they get automatically garbage collected, function could
    # initialize KDTrees for example)
    # or put the data of the KDtree on them so that they take ob the same memory space

    pg = np.full(shape=(*res_grid,2), fill_value=-1, dtype=int)
    pg[0,0] = idx2co(random.randrange(pnum), max_co) #get first patch

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

    print("generating patches")
    m_patches, max_co, idx2co = gen_patches_from_mask(img,mask)
    #build local neighbourdhood KDTree for pyramid level
    print("generating tree from patches")
    tree = sklearn.neighbors.KDTree(m_patches, metric='l2')
    return tree, mask, mask_center, idx2co

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

def mask_synthesize_pixel(level, example, seed, pyramid, target = None):
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

    target = mask_synthesize_pixel(start_level, example, seed, py)
    tas.append(target)
    for level in range(start_level+1,len(py)):
        print(f"\nstarting next level calculation: {level}\n")
        target = mask_synthesize_pixel(level, example, seed, py, target)
        tas.append(target)

    return target, tas

def normalize_picture(example0, max_pixels = 256*256):
    #max_pixels basically defines whats possible with the avialable
    # memory & CPU power 256x256 has proved to be effective on modern systems
    ex_pixels = np.prod(example0.shape[:2])
    scaling = 1.0
    if max_pixels < ex_pixels:
        px_ratio = ex_pixels/max_pixels
        scaling = 1/math.sqrt(px_ratio)
        print(f"resizing with scaling {scaling}")
        example = skimage.transform.rescale(example0, scaling,
        #example = skimage.transform.resize(example0, (256,256),
                                            anti_aliasing=True,
                                            multichannel=True,
                                            preserve_range=True)#.astype(np.uint8)
        #search_res = example.shape[:2]
    else: example = example0

    return example, scaling

def create_patch_params(example0, scaling,
                        overlap_ratio = 1/6, patch_ratio = 0.05):
    """patch_ratio = #size of patches in comparison with original
    """
    #TODO: define minimum size for patches
    res_patch2 = int(min(example0.shape[:2])*patch_ratio)
    res_patch2 = np.array([res_patch2]*2)
    res_patch = np.round(res_patch2*scaling).astype(int)
    overlap = np.ceil(res_patch*overlap_ratio).astype(int)
    #res_patch2 = np.round(np.array(res_patch)/scaling).astype(int)
    overlap2 = np.ceil(res_patch2*overlap_ratio).astype(int)

    return res_patch, res_patch2, overlap, overlap2

def synth_patch_tex(target, example0, k=5):

    example, scaling = normalize_picture(example0)
    res_target = target.shape[:2]
    res_patch, res_patch2, overlap, overlap2 = create_patch_params(example0, scaling)
    res_grid = np.ceil(res_target/(res_patch2 - overlap2)).astype(int)

    print(f"patch_size: {res_patch2}\ninitial scaling: {scaling}, ")

    #time.sleep(10.0)

    pg = synthesize_grid(example, res_patch, res_grid, overlap, tol=.1, k=k)
    pgimage = pg/pg.max((0,1))
    pgimage = np.dstack((pgimage,np.zeros(pgimage.shape[:2])))

    #transform pg coordinates into original source texture
    pg2 = np.round(pg / scaling).astype(int)

    target = transform_patch_grid_to_tex(None, res_patch2, pg2, example0,
                                         overlap2,
                                         use_quilting=True)

    target2 = transform_patch_grid_to_tex(None, res_patch, pg, example,
                                          overlap,
                                          use_quilting=True)

    return target[:res_target[0],:res_target[1]], target2, pgimage

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

    return target, mask, pxverts


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

def fill_area_with_texture(target, example0, 
                           verts=None, bmask = None, bounding_box = None):
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
    if bmask is None:
        mask = draw_polygon_mask(verts,target.shape[:2])
        bmask = mask[y0:y1,x0:x1]>0
    #bmask2 = mask[y0:y1,x0:x1]>0
    #area.boundary.buffer(100)

    print("synthesize texture")
    fill1, fill2, pgimg = synth_patch_tex(bbox, example0, k=1)
    if bounding_box is None: copy_img(target, fill1, (x0,y0), bmask)
    else: copy_img(target, fill1, (x0,y0), bmask)
    #import ipdb; ipdb.set_trace() # BREAKPOINT

    return target, bmask, fill1, fill2, pgimg

def check_memory_requirements(example, res_patch, maxmem = 1.0):
    patch_num=np.product(np.array(example.shape[:2]) - res_patch)
    ch_num = example.shape[2]
    data_memoryGB = patch_num*ch_num*example.itemsize*np.product(res_patch)*GB
    print(f"using approx. {data_memoryGB:2f} GB in RAM.")
    if data_memoryGB > maxmem:
        raise MemoryError("the algorithm would exceed the "
                          f"maximum amount of Memory: {data_memoryGB:2f} GB,; max: {maxmem}")
    return data_memoryGB

@timing
def prepare_tree(example0, lib_size, overlap_ratio, patch_ratio):
    example, scaling = normalize_picture(example0, lib_size)
    res_patch, res_patch0, overlap, overlap0 = create_patch_params(example0, scaling,
                                                                   overlap_ratio,
                                                                   patch_ratio)
    max_co = np.array(example.shape[:2]) - res_patch

    check_memory_requirements(example,res_patch, maxmem=1.0)
    data = create_patch_data(example, res_patch, max_co)
    tree = sklearn.neighbors.KDTree(data, metric='l2')
    return tree, res_patch, res_patch0, overlap, overlap0, max_co, scaling

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

def isolate_edge():
    """isolates an edge and copies the area around the edge
    in a separate canvas"""

    offset = 10

    n = [calculate_face_normal(f) for f in verts]

    #get a rectangle on the "inner" side of each edge
    #flip x & y for the cross product because of reversed coordinate order in numpy arrays
    #and image indexing

    v_ov1 = verts[0][3]-verts[0][0]
    v_ov2 = verts[0][2]-verts[0][1]

    edge_area = np.array((*e1,
                 e1[1]+normalized(v_ov2)*offset/math.sin(calc_angle_vec(-v1,v_ov2)),
                 e1[0]+normalized(v_ov1)*offset/math.sin(calc_angle_vec(v1,v_ov1))))

    y0,x0,y1,x1 = np.floor(shapely.geometry.Polygon(edge_area).bounds).astype(int)
    y1+=1;x1+=1
    edge_synth_target = target1[y0:y1,x0:x1].copy()
    #vi = normalized(np.cross(v1[::-1],(0,0,n[0])))[:2][::-1]

    #e1_i = e1 + vi*offset
    target1[skimage.draw.polygon(*edge_area.T)] *= (0.5,0.5,1.0,1.0)

    skimage.io.imshow_collection([edge_synth_target])
    #angle between x-axis and edge
    alpha = math.atan2(*v1)
    tmp = skimage.transform.rotate(edge_synth_target, alpha*180/math.pi, resize=True, order=3)
    skimage.io.imshow_collection([tmp])

    skimage.io.imshow_collection([target1])

    cy = np.round(math.sin(alpha)*(x1-x0)).astype(int)
    h = offset+5
    skimage.io.imshow_collection([tmp[cy-h:cy,:]])

#TODO: "functionalize" the below function.
# this means: calculate all the below things in individual matrices
# and numpy arrays
def transfer_patch_pixelwise(target, search_area0, 
                             yp,xp, patch_index,
                             edge_info,
                             fromtarget,
                             face_source,
                             sub_pixels = 1,
                             mask_sides = None,
                             mask = None,
                             pa = None):
    """This function takes an edge and pixel, and searches for the corresponding
    pixel at another edge.
    """
    e1,e2,v1,v2,e2_perp_left = edge_info
    for sub_pix in np.ndindex(sub_pixels,sub_pixels):
        sub_idx =  np.array(patch_index) + np.array(sub_pix)/sub_pixels
        coords = np.array((yp,xp)) + sub_idx - e1[0]
        c = coords.dot(v1)/(v1.dot(v1))#ortihonal projection of pixel on edge
        d = c*v1-coords
        #pixel on left or right side of edge?
        isleft = (v1[1]*d[0]-d[1]*v1[0])>0
        d_len = norm(d)

        if isleft:
            #calculate corresponding pixel at other edge in reverse direction
            # (this is why we use (1-c))
            p_e2 = (1-c)*(v2) + e2[0]
            px2_coords = p_e2 + d_len * e2_perp_left
            tmp = np.round(px2_coords).astype(int)
            if fromtarget:
                #copy pixel from target to the search index
                #TODO: use interpolation here (if edge lengths differ in length)
                search_area0[patch_index] = target[tuple(tmp)]
            else:
                mask_sides[patch_index]=1.0
                if check_inside_face(face_source, tmp):
                    if mask[patch_index]>0:
                         #copy pixel from generated patch back to target
                        target[tuple(tmp)] = pa[patch_index]
                #target[tuple(tmp)] = pa[patch_index]

def make_seamless_edge(e1,e2, target, example0, debug_level=0):
    (e1,verts1),(e2,verts2) = e1,e2
    v1 = e1[1]-e1[0]
    v2 = e2[1]-e2[0]

    #edge_perpendivular vectors pointing to the "left" of an edge
    e2_perp_left = normalized(v2[::-1]*(1,-1))

    #move along the edge and generate patches from information
    #from both sides of the edge
    lib_size=128*128
    overlap_ratio = 1/6
    patch_ratio = 0.1


    target_new = target.copy()
    (tree, res_patch,
     res_patch0, overlap,
     overlap0, max_co, 
     scaling) = prepare_tree(example0, lib_size, overlap_ratio, patch_ratio)

    edge_dir = normalized(v1)
    #TODO: iterate over the "longer" of the two edges. This way we do not
    #need to scale the patches? --> we also have to ransfer back so that
    #doesnt really make sense...  the only thin is:
    #it might be better to have the transoformation from the optimized patch
    #back to the second edge with fewer
    #"disturbances" meaning: it would be better to have the second edge the
    #shorter one. While it is not as important to have an
    #undisturbed view search patch
    step_width = min(res_patch0)*0.5
    for counter, i in enumerate(np.arange(0,norm(v1)+step_width,step_width)):
        # TODO: for the corners we need to search the faces and edges
        # connected to this corner to fill the patc with corresponding
        # pixels
        cy,cx = e1[0] + i*edge_dir #calculate center of patch
        yp,xp = np.round((cy,cx) - res_patch0/2).astype(int) #calculate left upper corner of patch

        #target_new[skimage.draw.circle(cy,cx,2)]=(1,0,0,1)
        search_area0 = target_new[yp:yp+res_patch0[0],
                                  xp:xp+res_patch0[1]].copy()
        for patch_index in np.ndindex(search_area0.shape[:2]):
            transfer_patch_pixelwise(target_new, search_area0, 
                                     yp,xp, patch_index,
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
        co_p0 = np.round(co_p / scaling).astype(int)
        ovs = np.r_[overlap0,overlap0]
        pa, pa0, ta0, mask = optimal_patch(search_area0, example0, res_patch0,
                                         ovs, co_p0, (yp,xp))        
        if debug_level>0: #for debugging
            search_area0[:,:,0:2]=(1,1)
            pa[:,:,1]=1
        
        #copy one side of the patch back to its respective face 
        #and also create a left/rght mask for masking the second part
        mask_sides = np.zeros(search_area0.shape[:2])
        for patch_index in np.ndindex(search_area0.shape[:2]):
            transfer_patch_pixelwise(target_new, search_area0, 
                                     yp,xp, patch_index,
                                     edge_info = (e1,e2,v1,v2,e2_perp_left),
                                     fromtarget = False,
                                     face_source = verts2,
                                     sub_pixels = 2,
                                     mask_sides = mask_sides,
                                     mask = mask,
                                     pa = pa)
        
        
        #TODO: copy only the part thats "inside" face 1
        mask_inside = np.zeros(search_area0.shape[:2])
        for patch_index in np.ndindex(search_area0.shape[:2]):
            coords = patch_index + np.array((yp,xp))
            mask_inside[patch_index] = check_inside_face(verts1,coords)
            
        #copy only the right side to its place
        #mask_right_optimal = mask_sides==0
        mask_right_optimal = np.minimum(mask_sides==0, mask>0)
        mask_right_optimal = np.minimum(mask_right_optimal, mask_inside>0)
        copy_img(target_new, pa, (xp,yp), mask_right_optimal)

        #if counter == 2:
        if False:#debug_level>0:#counter == 2:
            #patch_from_data = data[new_idx].reshape(*res_patch,4)
            skimage.io.imshow_collection([search_area0, pa, mask_sides])
            #skimage.io.imshow_collection([search_area0, search_area, target1,
            #                      patch_from_data, pa,pa0,ta0])
            break
        
    return target_new

def check_inside_face(polygon, point):
        face = shapely.geometry.Polygon(polygon)
        return face.contains(shapely.geometry.Point(*point))

if __name__ == "__main__":
    #example0 = example = skimage.io.imread("textures/3.gif") #load example texture
    example0 = skimage.io.imread("textures/rpitex.png")
    example0 = example0/255
    #example0 = skimage.transform.resize(example0, (500,1000))
    example0 = skimage.transform.rescale(example0, 0.3, multichannel=True)
    #example0 = example = skimage.io.imread("RASP_03_05.png") #load example texture
    #TODO: more sophisticated memory reduction techniques (such as
    # a custom KDTree) This KDTree could be based on different, hierarchical
    # image resolutions for the error-search

    #lower brightnss of bounding box for debugging ppurposes
    if True:
        np.random.seed(10)
        random.seed(50)#2992)#25 is chip + original img
        target1, _, verts = generate_test_target_with_fill_mask(example0)
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
        print("alter next edge")
        target_new = make_seamless_edge((e1,verts[0]),(e2,verts[1]), 
                                        target_new, example0)
    
    skimage.io.imshow_collection([target_new, target1])
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
    