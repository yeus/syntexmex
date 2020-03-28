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
"""

import bpy
import bmesh
import numpy as np
import itertools
import skimage
import skimage.io
import skimage.transform
import networkx as nx
import logging
import pickle
logger = logging.getLogger(__name__)

#def norm(x): return np.sqrt(x.dot(x))
def norm(x): return np.sqrt((x*x).sum(-1))
#need to be transposed for correct ultiplcation along axis 1
def normalized(x): return (x.T /norm(x)).T

def debug_image(img, name = None):
    if img.shape[2]<4:#add alpha channel
        img = np.pad(img,((0,0),(0,0),(0,1)),constant_values=1)
    
    new_tex = bpy.data.images.new("debug", width=img.shape[1], 
                                       height=img.shape[0])
    new_tex.pixels[:] = img.flatten()

def create_bmesh_from_active_object():
    if bpy.context.active_object.mode == 'OBJECT':
        ob = bpy.context.selected_objects[0]
        me = ob.data
        bm = bmesh.new()   # create an empty BMesh
        bm.from_mesh(me)   # fill it in from a Mesh
    else:
        ob = bpy.context.edit_object
        me = ob.data
        bm = bmesh.from_edit_mesh(me)

    # the next step makes operations possible in non-edit mode
    bm.faces.ensure_lookup_table()
    return ob,bm

#TODO: find_uv_origin(uv): #so that we know where and how
# to start the texture synthesis
def find_uv_origin(uv):
    """ find out orientation of the uv face by 
     calculating the normal in the direction of the loops
     we want the "left" of the two uv vertices
     to be the starting point. and that depends on the
     orientation of the uv face"""
    uv_n = (uv[1] - uv[0]).cross(uv[2] - uv[1])
    if uv_n < 0: #means loop direction is clockwise
        origin = uv[1]
    else: #means loop direction is counter clockwise
        origin = uv[0]

def find_minmax(uv):
    y_max,idx = max((v[1],i) for i,v in enumerate(uv))
    x_max,idx = max((v[0],i) for i,v in enumerate(uv))

    y_min,idx = min((v[1],i) for i,v in enumerate(uv))
    x_min,idx = min((v[0],i) for i,v in enumerate(uv))
    return x_min,x_max,y_min,y_max

def create_initial_image(res):
    import skimage as skim
    #TODO: add option to remove "random shapes"
    tmp_img,_ = random_shapes((res[1],res[0]),
                            max_shapes=20,
                            intensity_range=((100, 255),))

    img = skim.util.img_as_float(tmp_img)
    #add padding
    img = np.pad(img,((res[1],res[1]),(res[0],res[0]),(0,0)),constant_values=0)
    return img

def add_uvs(img, xy):
    # draw uvs in the image for debugging purposes:
    for x,y in xy:
        rr,cc = skim.draw.circle(y, x, 5)
        img[rr,cc,:]=(0,0,1)
        rr,cc = skim.draw.circle(y, x, 0.5)
        img[rr,cc,:]=(1,0,0)

def add_face_shadow(img, xy):
    x,y = zip(*xy)
    rr,cc = skim.draw.polygon(y, x)
    img[rr,cc,2] = 0
    img = np.clip(img,0.0,1.0)

def copy_img(target, src, pos, mask=None):
    """
    copy image src to target at pos
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


def create_mask(vecs):
    #TODO: create a "soft" mask
    #rr,cc = skim.draw.polygon(y, x)
    #mask = np.zeros_like(img)
    #mask[rr,cc,0] = 0.5
    mask = skim.draw.polygon2mask(img.shape[:2], vecs)
    return mask

def draw_face_outline(img, uv_p, uv_vp):
    #color only the corners
    abs = ((0,0),(0,1),(1,0),(1,1))
    for a,b in abs:
        #corners
        cbase = uv_p[0] + a * uv_vp[0] + b * uv_vp[3]
        rr,cc = skim.draw.circle(cbase[1], cbase[0], 5)
        target[rr,cc,:3]=(0,0,1)
        
        #edge1
        a_length = mh.norm(uv_vp[0])
        for i in np.arange(0,a_length,0.3):
            a = i/a_length
            b = 0
            base = uv_p[0] + a * uv_vp[0] + b * uv_vp[3]
            idx = base.astype(int)
            img[idx[1],idx[0],:3]=(0,1,0)
            #import ipdb; ipdb.set_trace() # BREAKPOINT

        b_length = mh.norm(uv_vp[3])
        for i in np.arange(0,a_length,0.3):
            a = 0
            b = i/a_length
            base = uv_p[0] + a * uv_vp[0] + b * uv_vp[3]
            idx = base.astype(int)
            img[idx[1],idx[0],:3]=(0,1,0)
            #import ipdb; ipdb.set_trace() # BREAKPOINT

        target[cbase[1].astype(int), cbase[0].astype(int)
                ,:3]=(1,0,0)

def draw_triangle_outline(img, uv_p, uv_vp):
    a_length = mh.norm(uv_vp[0])
    b_length = mh.norm(uv_vp[1])
    for b in np.arange(0,b_length,0.5):
        #for a in np.arange()
        a = 1-b/b_length
        b = b/b_length
        x = uv_p[0] + a * uv_vp[0] + b * uv_vp[3]
        img[x[1].astype(int),x[0].astype(int),:3]=(1,1,0)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);

def PointInTriangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0);
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0);

    return not (has_neg and has_pos);

def idx_sanitize(co, img):
    x_max = img.shape[:2][::-1]
    x_min = (0,0)
    z = np.min([x_max,co], axis=0)
    return np.max([z,x_min], axis=0)

def idx_check(co, img):
    x_max = img.shape[:2][::-1]
    x_min = np.array((0,0))
    return np.all((co<x_max, co>=x_min))

def init_face_map(img, bm):
    #TODO: get rid of empty pixels
    #TODO: make algorithm more efficient
    for face in bm.faces:#[:1]:
        uv = np.array([l[uv_layer].uv for l in face.loops])
        uv_p = uv * res[:2] #transform to pixel space

        #get four uv edge vectors:
        uv_v = np.array([uv[1]-uv[0],
                        uv[2]-uv[1],
                        uv[3]-uv[2],
                        uv[0]-uv[3]])


        #TODO: do the followgin for both face triangles
        tris = [(0,1,2),(2,3,0)]
        img = face_map
        for tr in tris:
            base = uv_p[tr[1]]
            uv_v = np.array([uv[tr[0]]-uv[tr[1]],
                             uv[tr[2]]-uv[tr[1]]])

            #transform to pixel space
            uv_vp = uv_v * res[:2]
            
            #draw patches
            a_length = mh.norm(uv_vp[0])
            b_length = mh.norm(uv_vp[1])
            st_len = 0.8
            for b in np.arange(0,b_length,st_len):
                #for a in np.arange()
                b_ = b/b_length
                c_ = a_length * (1-b_)
                #import ipdb; ipdb.set_trace() # BREAKPOINT   
                for a_ in np.arange(0,c_,st_len):
                    a_ = a_/a_length
                    x = base + a_ * uv_vp[0] + b_ * uv_vp[1]
                    x = x.astype(int)
                    #if x[1]==300:
                    #    import ipdb; ipdb.set_trace() # BREAKPOINT
                    if idx_check(x, img):
                        img[x[1],x[0]]=face.index
        
    #draw_face_outline(target, uv_p, uv_vp)
    #draw_triangle_outline(target, uv_p, uv_vp)  
          
    #import ipdb; ipdb.set_trace() # BREAKPOINT  

def empty_copy(img, dtype):
    #return np.zeros(img.shape[:2], dtype = dtype)
    return np.full(img.shape[:2], -1, dtype = dtype)


def blimage2array(blender_source):
    res = np.array((blender_source.size[1],blender_source.size[0]))
    return np.array(list(blender_source.pixels)).reshape(*res,4)

def init_texture_buffers(example_image, target_texture, example_scaling):
    # numpy handles shapes in a different way then images in blender,
    # because of this, the convention, when indexing looks like this:
    # shape = (y from bottom, x from left, alpha)
    res_ex = np.array((example_image.size[1],example_image.size[0]))
    nt =  target_texture
    res = np.array((nt.size[1], nt.size[0])) 
    # define numpy buffers for image data access
    target = np.array(list(nt.pixels)).reshape(*res,4) # create an editable copy
    example = np.array(example_image.pixels).reshape(*res_ex,4)
    example = skimage.transform.rescale(example, example_scaling, 
multichannel=True)
    
    return example, target

    #import ipdb; ipdb.set_trace() # BREAKPOINT

# TODO: calculate face area and scale example
# accordingly  

#TODO:
#find center, so that we can either start the 
#algorithm in the center or extend a little bit beyond the 
# borders of the face:
#uv_center = np.sum(uv, axis=0)/4

"""
face_map = empty_copy(target, int)
init_face_map(face_map, bm) 

#convert facemap into image
num_faces = len(bm.faces)
#create random colors for faces (+1 because index starts with 0)
rand_colors = np.random.rand(num_faces+1,3)
map2 = np.array([rand_colors[px] if px != -1 else (0,0,0) for px in face_map.flatten()])
shape = (*target.shape[0:2],3)
map2 = map2.reshape(shape)
#if display_face_map:
target[:,:,0:3] = map2
#import ipdb; ipdb.set_trace() # BREAKPOINT
"""

#debug_img(tmp0)
#import ipdb; ipdb.set_trace() # BREAKPOINT

#randomly copy patch to target image
#coords = np.random.rand(2)*res[:2]/2 + res[:2]/4
#copy_img(target, patch, coords.astype(int))

#TODO: get rid of empty pixels
#TODO: make algorithm more efficient

#TODO: make a map of "remaining" pixels which did not get painted
# to get rid of holes

"""
img = target
for face in bm.faces[:]:
    uv = np.array([l[uv_layer].uv for l in face.loops])
    uv_p = uv * res[:2] #transform to pixel space

    #get four uv edge vectors:
    uv_v = np.array([uv[1]-uv[0],
                    uv[2]-uv[1],
                    uv[3]-uv[2],
                    uv[0]-uv[3]])


    #TODO: do the followgin for both face triangles
    tris = [(0,1,2),(2,3,0)]
    for tr in tris:
        base = uv_p[tr[1]]
        uv_v = np.array([uv[tr[0]]-uv[tr[1]],
                         uv[tr[2]]-uv[tr[1]]])

        #transform to pixel space
        uv_vp = uv_v * res[:2]
        
        #draw patches
        a_length = mh.norm(uv_vp[0])
        b_length = mh.norm(uv_vp[1])
        st_len = 20.0
        for b in np.arange(0,b_length,st_len):
            #for a in np.arange()
            b_ = b/b_length
            c_ = a_length * (1-b_)
            #import ipdb; ipdb.set_trace() # BREAKPOINT   
            for a_ in np.arange(0,c_,st_len):
                a_ = a_/a_length
                x = base + a_ * uv_vp[0] + b_ * uv_vp[1]
                patch = random.choice(patches)
                coords = mh.vec((x[0]-res_patch[0]*0.5,
                                 x[1]-res_patch[1]*0.5)).astype(int)
                #if all(coords == (  -8, 1013)):
                    #import ipdb; ipdb.set_trace() # BREAKPOINT
                copy_img(target, patch, coords)
                #target[coords[1],coords[0],:3]=(0.5,0.5,1.0)
                #img[x[1].astype(int),x[0].astype(int)]=patch
"""

def get_uv_levelset(uvs):
    from shapely.geometry import Polygon, Point
    
    uv_p = uvs * res[:2] #transform to pixel space
    face = Polygon(uv_p)
    boundary = face.boundary
    face_container = face.buffer(+10.0) #add two pixels on the container
    bbox = face_container.bounds
    minx, miny, maxx, maxy = bbox_px = np.round(np.array(bbox)).astype(int)
    w,h = maxx - minx, maxy-miny

    def distance(x,y):
        d = boundary.distance(Point(x,y))
        if face.contains(Point(x,y)): return d
        else: return -d 

    bbcoords = itertools.product(range(miny,maxy), range(minx, maxx))
    levelset = np.array([distance(x,y) for y,x in bbcoords]).reshape(h,w)
    #normalize levelset:
    #levelset = np.maximum(levelset/levelset.max(),0.0)
    return levelset, bbox_px

#get all UVs per face:

def get_uvs(bm):
    uvs = {}#defaultdict(list)
    for face in bm.faces:#[:1]:
        uvs[face.index] = get_face_uvs(face,bm)
    return uvs

def get_face_uvs(face, bm):
    uv_layer = bm.loops.layers.uv['UVMap']
    uv = np.array([l[uv_layer].uv for l in face.loops])
    return uv

def get_edge_uvs(edge, bm):
    uv_layer = bm.loops.layers.uv['UVMap']
    #if len(edge.link_loops) == 2:
    l = edge.link_loops #the two "opposite loops"
    ln = [l.link_loop_next for l in edge.link_loops] #for the next uv
    uv_edges = np.array([[l[uv_layer].uv for l in lln] for lln in zip(l,ln)])
    #uv_edge2 = np.array([l[uv_layer].uv for l in [l2,l2n]])
    return uv_edges


def loops_connected(edge, bm):
    """check if the uv coordinates of an edge are the
    same for both connected faces"""
    uv_edges = get_edge_uvs(edge, bm)
    if len(uv_edges)<2:
        return False
    else:
        if np.all(uv_edges[0]==uv_edges[1][::-1]): return True
        else: return False
        
def is_border_edge(edge, bm):
    if len(edge.link_loops)<2:
        return True
    else: return False

def generate_edge_loop_uvs(bm_edges, res, bm):
    """generates an edge_info structure that can be used
    as input to make seamless edges"""
    edge_uvs = [get_edge_uvs(e, bm) for e in bm_edges] #get uvs
    #[len(e) for e in edge_uvs]
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    #switching x and y coordinates to adapt to numpy column-row convention
    edge_uvs = np.array(edge_uvs)[...,::-1]*res + (-0.5,-0.5) #switch xy to numpy yx convention and transform into pixel space
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    face_uvs = np.array([(e.link_faces[0].index,
                 e.link_faces[1].index) for e in bm_edges])
    #import ipdb; ipdb.set_trace()
    #for face its a little more complicated switching the coordinates
    #as there can have different number of vertices and thus
    # can not be put into a homogenous numpy array
    #face_uvs = [[f[...,::-1]*res + (-0.5,-0.5) for f in fs] for fs in face_uvs]
    #face_uvs = [np.array(face_uvs)[...,::-1]*res + (-0.5,-0.5)]
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    edge_infos1 = tuple(zip(edge_uvs[:,0,:,:],face_uvs[:,0]))
    edge_infos2 = tuple(zip(edge_uvs[:,1,:,:],face_uvs[:,1]))
    return edge_infos1,edge_infos2

def prepare_uv_synth_info(example, 
                          target, bm, 
                          patch_ratio, 
                          libsize):
    logger.info("generate uv synth info")
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    #generate a list of connected faces
    bm.edges.index_update()
    connected_edges = [e for e in bm.edges if loops_connected(e, bm)]
    #build a graph of connected faces
    connected_faces = [(e.link_faces[0].index,e.link_faces[1].index, e.index) 
                        for e in connected_edges]
    G = nx.Graph()
    G.add_weighted_edges_from(connected_faces, weight='index')
    islands = list(nx.connected_components(G))
    
    res = np.array(target.shape[:2])
    #res_ex 
        
    #import ipdb; ipdb.set_trace() # BREAKPOINT

    #build a list of edge pairs
    #iterate through edges
    #https://b3d.interplanety.org/en/learning-loops/
    bm.edges.index_update()
    continuous_edges = [edge for edge in bm.edges if not is_border_edge(edge,bm)]
    unconnected_edges = [edge for edge in continuous_edges if not 
                         loops_connected(edge, bm)] 

    #import ipdb; ipdb.set_trace() # BREAKPOINT
    edge_infos1, edge_infos2 = generate_edge_loop_uvs(unconnected_edges, 
                                                      res, bm)
    uv_info = {
            "target":target,
            "example":example,
            "patch_ratio":patch_ratio,
            "libsize":libsize,
            "face_uvs":get_uvs(bm),
            "islands":islands,
            "edge_infos":list(zip(edge_infos1, edge_infos2))
            }
    
    #uncomment, if picklig is needed for debugging the senthesis algorithm
    #needs to be debugged outside of blender
    if True: 
        logger.info("saving pickle")
        with open('uv_test_island.pickle', 'wb') as handle:
            pickle.dump(uv_info, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    return uv_info

    
"""for face in bm.faces[:1]:
    uv = np.array([l[uv_layer].uv for l in face.loops])
    
    import ipdb; ipdb.set_trace() # BREAKPOINT
"""
#for uvs in list(get_uvs(bm).values())[:]:
#    #iterate through each edge
#    import ipdb; ipdb.set_trace() # BREAKPOINT


#TODO: find out which edges are already "connected", because

#make UVs seamless

#for these we don't need to copy the pixels  

#TODO: for directly connected faces we can omit the whole
#seamless procedure and render the entire "metaface" as a whole
#and thn concentrate on indiviual edges oly afterwards

#debug_image(cospxs)
#debug_image(f2)
#debug_image(f1)
#import ipdb; ipdb.set_trace() # BREAKPOINT


"""
    #get four uv edge vectors:
    uv_v = np.array([uv[1]-uv[0],
                    uv[2]-uv[1],
                    uv[3]-uv[2],
                    uv[0]-uv[3]])
"""

#find points on the "outside" of uv_edges: which correspond to certain
#points in other faces according to some rule (probably try to resemble a
# straight line as much as possible)


#debug_image(face_map)

#bmesh.update_edit_mesh(me, True)


