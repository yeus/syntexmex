#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
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

and some more inspiration from:
- wfc synthesis algorithm

TODO: optionally in the future:
- Graphcut Textures: Image and Video Synthesis Using Graph Cuts [Kwatra, Schödl]
"""

__author__ = "Thomas Meschede / Thomas Meschede Rose"
__copyright__ = "Copyright (C) 2020 Thomas Rose Meschede"
__license__ = "Proprietary"
__version__ = "1.0"

import random
import numpy as np
import skimage
import skimage.io
import skimage.transform
import threading
#import gc
import math
import functools
sign = functools.partial(math.copysign, 1) # either of these
import logging
logger = logging.getLogger(__name__)
try:
    from . import tex_synthesize as ts
except ImportError:
    import tex_synthesize as ts
#import tex_synthesize as ts
import pickle


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

def tqdm(iterator, *args, **kwargs):
    return iterator

GB = 1.0/1024**3 #GB factor

def normalized_synthmap(synthmap, example):
    return synthmap/(*example.shape[:2][::-1],1)

@ts.timing
def synthesize_textures_on_uvs(synth_tex=False,
                               seamless_UVs=False,
                               msg_queue=None,
                               stop_event=None,
                               edge_iterations=0,
                               *argv, **kwargs):
    """
    msg_queue lets the algorithm share intermediate steps
    when using threaded calculations (queue.Queue)
    """
    target = kwargs['target']
    example = kwargs['example']
    patch_ratio = kwargs['patch_ratio']
    libsize = kwargs.get('libsize',128*128)
    face_uvs = kwargs['face_uvs']
    islands = kwargs['islands']
    edge_infos = kwargs['edge_infos']
    seed = kwargs.get('seed_value', 0)
    logger.info(f"seed: {seed}")
    if msg_queue is None: msg_queue=False
    if stop_event is None: stop_event=threading.Event()
    
    #set random seed for algorithm
    np.random.seed(seed)
    random.seed(seed)
    
    ta_map = None

    #TODO: check whether we have "left or right" sided coordinate system
    
    if synth_tex: #generate initial textures
        #TODO: make sure all islands are taken into account
        logger.info("synthesize uv islands")
        res = target.shape[:2]
        for island in islands:
            island_uvs = [face_uvs[i] for i in island]
            island_uvs_px = np.array([uv[...,::-1] * res[:2] for uv in island_uvs])
            #get a boundingbox for the entire island
            isl_mins = np.array([isl_px.min(axis=0) for isl_px in island_uvs_px])
            ymin,xmin = isl_mins.min(axis=0).astype(int)#-(1,1)
            ymin,xmin = max(ymin,0),max(xmin,0)
            isl_mins = np.array([isl_px.max(axis=0) for isl_px in island_uvs_px])
            ymax,xmax = isl_mins.max(axis=0).astype(int)#+(1,1)
            ymax,xmax = min(ymax,res[0]),min(xmax,res[1])
        
            #add .5 so that uv coordinates refer to the middle of a pixel
            # this has to be done after the "mins" where found
            island_uvs_px = [isl + (-0.5,-0.5) for isl in island_uvs_px]
            
            island_mask = np.zeros(target.shape[:2])
            for uvs in island_uvs_px:
                island_mask[skimage.draw.polygon(*uvs.T)]=1.0
            island_mask = island_mask[ymin:ymax,xmin:xmax]>0

            target, ta_map = ts.fill_area_with_texture(target, example, ta_map,
                                      patch_ratio=patch_ratio, libsize = libsize,
                                      bounding_box=(ymin,xmin,ymax,xmax),
                                      mask = island_mask)
            if msg_queue: msg_queue.put((target,normalized_synthmap(ta_map,example)))

    if stop_event.is_set(): 
        logger.info("stopping_thread")
        return

    if seamless_UVs:
        tree_info = None
        if ta_map is None:
            ta_map = np.ones([*target.shape[:2],3])*-1
        for i,(e1,e2) in enumerate(edge_infos):
            logger.info(f"making edge seamless: #{i}")
            #TODO: add pre-calculated island mask to better find "valid" uv pixels
            edge1 = e1[0],face_uvs[e1[1]][:,::-1]*target.shape[:2]
            edge2 = e2[0],face_uvs[e2[1]][:,::-1]*target.shape[:2]
            target, ta_map, tree_info = ts.make_seamless_edge(edge1, edge2, 
                                           target, example, ta_map,
                                           patch_ratio, libsize, 
                                           tree_info=tree_info,
                                           debug_level=0)
            if msg_queue: msg_queue.put((target,normalized_synthmap(ta_map,example)))
            if (edge_iterations != 0) and (i >= edge_iterations): break 
            if stop_event.is_set(): 
                logger.info("stopping_thread")
                return
        #debug_image(target2)
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        
    return (target,
            normalized_synthmap(ta_map,example),
            ta_map)

def check_face_orientation(face):
    edge_vecs = np.roll(face,1,0)-face
    return np.cross(np.roll(edge_vecs,1,0),edge_vecs)

def paint_uv_dots(faces, target):
    for f in faces.values():
        for v in f[:,::-1]*target.shape[:2]:
            #v=v[::-1]
            target[skimage.draw.circle(v[0],v[1],2)]=(1,0,0,1)
  
def reconstruct_synthmap(synthmap,example,mode="coordinates"):
    """
    TODO: reconstruct from multiple examples as well (third channel in ta_map)
    
    modes: "coordinates", "normalized"
    """
    if mode=="coordinates":
        return example[synthmap[:,:,1],synthmap[:,:,0]].copy()
    elif mode=='normalized':
        synthmap = synthmap * (*example.shape[:2][::-1],1)
        synthmap = synthmap.astype(int)
        return example[synthmap[:,:,1],synthmap[:,:,0]].copy()
    
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    logging.getLogger('tex_synthesize').setLevel(logging.INFO)

    #logging.get
    
    with open('uv_test_island.pickle', 'rb') as handle:
            uv_info = pickle.load(handle)
             
    #skimage.io.imshow_collection([uv_info["target"],uv_info["example"]])
    
    target=uv_info['target']
    example=uv_info['example']
    #skimage.io.imshow_collection([uv_info['target']])
    #skimage.io.imshow_collection([example])
    #paint_uv_dots(uv_info['face_uvs'],target)
    
    
    ta, ta_map1, ta_map2 = synthesize_textures_on_uvs(synth_tex=False,
                                        seamless_UVs=True,
                                        edge_iterations=0,
                                        **uv_info)
    logger.info("finished test!")
    skimage.io.imshow_collection([target, ta_map1, ta_map2])
    
    skimage.io.imsave("test.png",ta_map1)
    
    ta_map_recon = ta_map1*(*example.shape[:2],1)
    ((ta_map_recon-ta_map2)**2).sum() #test the rounding-error (its pretty good)
    
    ta_map = ta_map2.astype(int)
    
    conv = reconstruct_synthmap(ta_map1,example, mode="normalized")
    skimage.io.imshow_collection([ta,conv, ((ta-conv)**2)[...,:3]])
    mask = np.all(ta_map1[:,:,:2]>0,axis=2)
    edge_seams = ts.copy_img(ta_map1.copy(),conv[:,:,:3],pos=(0,0),mask=mask)
    skimage.io.imshow_collection([target,ta, edge_seams])

    #uv_info['edge_infos'][0]

    #faces = uv_info['island_uvs']
    
    #[check_face_orientation(f) for f in faces]
    
