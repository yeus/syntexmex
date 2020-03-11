"""
TODO: 
    
tex
synthmex
mesh
example
variation
automatic

symextex
sytexmex
symex
temexa
exmegen



SynbyX syntyx

gensyntex

synexis -> taken

current: syntexmex

- plugin for texture synthesis on 3d meshes and removing edge seams

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
"""

import importlib
import bpy
import os
import sys
import logging
import numpy as np
import threading, queue, time
import functools
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
To set up the logging module to your needs, create a file $HOME/.config/blender/{version}/scripts/startup/setup_logging.py (this is on Linux, but you’re likely a developer, so you’ll know where to find this directory). If the folder doesn’t exist yet, create it. I like to put something like this in there:

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)8s %(name)s %(message)s')

for name in ('blender_id', 'blender_cloud'):
    logging.getLogger(name).setLevel(logging.DEBUG)

def register():
    pass
"""


#
#blend_dir = os.path.dirname(bpy.data.filepath)
# if blend_dir not in sys.path:
#   sys.path.append(blend_dir)
# temporarily appends the folder containing this file into sys.path
#main_dir = os.path.dirname(bpy.data.filepath) #blender directory
#sys.path.append(main_dir)
main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib') #addon dir "__init__.py" + lib
sys.path.append(main_dir)
logger.info(f"add library directory to syntexmex: {main_dir}")

from . import uv_prepare as up
importlib.reload(up)
from . import uv_synthesize as us
importlib.reload(us)
importlib.reload(us.ts)

__author__ = "yeus <Thomas.Meschede@web.de>"
__status__ = "test"
__version__ = "0.9"
__date__ = "2020 Feb 29th"

#TODO: fill this out
bl_info = {
    "name": "Syntexmex",
    "author": "yeus <Thomas.Meschede@web.de",
    "version": (0, 0, 3),
    "blender": (2, 82, 0),
    "location": "TODO View3D > Add > Mesh > Add Pipes",
    "description": "Generate example-base seamless textures",
    "warning": "",
    "wiki_url": "http://yeus.gitlab.io",
    "category": "UV",
    "support": "COMMUNITY",
}

#from operator import *

# TODO: remove tabs
# https://blender.stackexchange.com/questions/97502/removing-tabs-from-tool-shelf-t-key/97503#97503

def multiline_label(layout,text):
    for t in text.split("\n"):
        layout.label(text=t)

class syntexmex(bpy.types.Operator):
    """This operator synthesizes texture in various ways on UV textures"""
    bl_idname = "texture.syntexmex"
    bl_label = "Synthesize Operations on UV Textures"
    bl_category = 'syntexmex'
    bl_options = {'REGISTER', 'UNDO'}
    
    patch_size: bpy.props.FloatProperty(
        name="Patch Size Ratio",
        description="Set width of patches as a ratio of shortest edge of an image",
        min=0.0,
        max=0.5,
        default=0.1,
        precision=3,
        step=0.1
    )
    example_image: bpy.props.StringProperty()
    example_scaling: bpy.props.FloatProperty(
        name="Example Scaling",
        description="""Scale Example to a certain size which will be used
to generate the texture""",
        min=0.0,
        max=1.0,
        default=1.0,
        precision=3,
        step=1
    )
    target_image: bpy.props.StringProperty()
    target_resolution: bpy.props.IntVectorProperty(name="synth-res.",
                                                   size=2,
                                                   min=0,
                                                   max=10000,
                                                   default=(1024,1024))
    synth_tex: bpy.props.BoolProperty(name="synthesize textures")
    seamless_UVs: bpy.props.BoolProperty(name="seamless UV islands")
    libsize: bpy.props.IntProperty(name="patch library size")
    seed_value: bpy.props.IntProperty(
            name="Seed Value",
            description="Seed value for predictable texture generation",
            default = 0
            )
    replace_material: bpy.props.BoolProperty(name="replace material",
                                             default=False)

    _timer = None

    @classmethod
    def poll(cls, context):
        #TODO: this operator should also be able to execute in other
        # windows as well
        return context.space_data.type in {'VIEW_3D'}
    
    def run_algorithm(self, context):
        logger.info("start synthesizing algorithm")
        #scene = context.scene
        
        #TODO: select specific object in menu
        obj, bm = up.create_bmesh_from_active_object()
        uv_layer = bm.loops.layers.uv['UVMap']
        
        logging.info(self.example_image)
        logging.info(self.target_image)
        
        example_image = bpy.data.images[self.example_image]
        if self.target_image is "":
            self.target_image = 'synthtarget'
            
        bpy.data.images.new(self.target_image,
                            self.target_resolution[0],
                            self.target_resolution[1],
                            alpha=False,float_buffer=True)
        self.target = bpy.data.images[self.target_image]
        
        ta_width,ta_height=self.target.size
        self.ta_map = bpy.data.images.new("ta_map",ta_width,ta_height,
                                          alpha=False,float_buffer=True)
        #BlendDataImages.new(name, width, height, alpha=False, float_buffer=False, stereo3d=False, is_data=False, tiled=False)
        
        examplebuf, targetbuf = up.init_texture_buffers(example_image,
                                                        self.target, self.example_scaling)
                
        self.msg_queue = queue.Queue()                       
        args = (examplebuf,
                targetbuf,
                bm,
                self.patch_size, 
                self.libsize)
        kwargs = dict()
        
        uv_info = up.prepare_uv_synth_info(*args,**kwargs)
        uv_info['seed_value']=self.seed_value
        self.algorithm_steps=self.synth_tex+len(uv_info['edge_infos']*self.seamless_UVs)
        context.scene.syntexmexsettings.synth_progress=0.01
        self._killthread = threading.Event()
        synthtex = functools.partial(us.synthesize_textures_on_uvs,
                                        synth_tex=self.synth_tex,
                                        seamless_UVs=self.seamless_UVs,
                                        msg_queue=self.msg_queue,
                                        stop_event=self._killthread,
                                        **uv_info)
        self._thread = threading.Thread(target = synthtex,
                                        daemon=True,
                                        args = [],
                                        kwargs = {})
        self._thread.start()
      
    def write_images(self,target,ta_map=None):
        # Write back to blender image.
        self.target.pixels[:] = target.flatten()
        self.target.update()
        
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        
        if ta_map is not None:
            ta_map = np.dstack((ta_map,np.ones(ta_map.shape[:2])))
            self.ta_map.pixels[:] = ta_map.flatten()
            self.ta_map.update()  

        logger.info("synced images!")
            
    def execute(self, context):
        def testworker(conn):
            for i in range(10):
                time.sleep(1.0)
                logger.info(f"working thread at {i}")
                conn.put(i)
            logger.info("working thread finished!")        
        
        self.run_algorithm(context)
        
        # start timer to check thread status
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        self._region = context.region
        self._area = context.area
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        #TODO: https://stackoverflow.com/questions/21409683/passing-numpy-arrays-through-multiprocessing-queue for passing images from the other (sub) process
        if event.type in {'ESC'}:#{'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            logger.info("thread was cancelled!")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # change theme color, silly!
            if self._thread.is_alive():
                self.receive(context)
                #print(".", end = '')
            else:
                logger.info("thread seems to have finished!")
                self.receive(context)
                self.ta_map.pack()
                self.target.pack()
                if self.replace_material:
                    bpy.ops.texture.syntexmex_pbr_texture(
                        synth_map=self.ta_map.name, 
                        source_material=context.scene.syntexmexsettings.source_material.name
                        )
                    
                self.cancel(context)
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def receive(self, context):
        try:
            msg = self.msg_queue.get_nowait()
        except queue.Empty:
            msg = None
        else:
            self.msg_queue.task_done()
            logger.info(f"received a new msg!!")
        if msg is not None:
            target,ta_map = msg
            self.write_images(target,ta_map)
            context.scene.syntexmexsettings.synth_progress += 1.0/self.algorithm_steps
            self._region.tag_redraw()
            self._area.tag_redraw()
        return msg

    def cancel(self, context):
        logger.info("cleaning up timer!")
        self._killthread.set()
        context.scene.syntexmexsettings.synth_progress=0.0
        self._region.tag_redraw()
        self._area.tag_redraw()
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
    
class clear_target_texture(bpy.types.Operator):
    """Clear target texture to make it black"""
    bl_idname = "texture.clear_target_texture"
    bl_label = "Clear target texture (black)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        logger.info("clear texture")
        ta = context.scene.syntexmexsettings.target_image
        
        target = np.zeros((ta.size[1], ta.size[0],4))
        target[...,3]=1.0
        
        ta.pixels[:] = target.flatten()
        ta.update()
        ta.pack()
        
        return {'FINISHED'}

def calc_syntexmex_info(ex_img, props):
    res_ex = np.array((ex_img.size[1],ex_img.size[0])) * props.example_scaling
    if props.target_image:
        ta_img = props.target_image
        res_ta = np.array((ta_img.size[1],ta_img.size[0]))
    else:
        res_ta = np.array((props.target_resolution[1],
                           props.target_resolution[0]))
    scaling = us.ts.calc_lib_scaling(res_ex, props.libsize)
    (res_patch, res_patch0,
    overlap, overlap0) = us.ts.create_patch_params2(res_ex,
                                             scaling,
                                             1/6.0, props.patch_size)
    #mem_reqs = tu.ts.check_memory_requirements2(res_ex,
    #                        res_patch, ch_num, )
    mem_reqs = 2*us.ts.calculate_memory_consumption(
          res_ex*scaling,
          res_patch,
          ch_num = ex_img.channels,
          itemsize = 8) #itemsize= 8 comes from the floatingpoint size of 8 bytes in numpy arrays

    maxmem=us.ts.get_mem_limit()

    return (res_ex,res_ta,scaling,res_patch, res_patch0,
            overlap, overlap0, mem_reqs, maxmem)

#TODO: ame everything according to here:
#https://wiki.blender.org/wiki/Reference/Release_Na.otes/2.80/Python_API/Addons
class syntexmex_panel(bpy.types.Panel):
    bl_label = "Syntexmex Configuration"
    bl_idname = "SYNTEXMEX_PT_syntexmex_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'syntexmex'
    #bl_context = "tool"

    def copy_to_operator(self, op, settings):
        op.example_image = settings.example_image.name
        if settings.target_image:
            op.target_image = settings.target_image.name 
        op.example_scaling = settings.example_scaling  
        op.libsize = settings.libsize 
        op.patch_size = settings.patch_size
        op.synth_tex=True      
        op.seamless_UVs=True
        op.seed_value=settings.seed_value
        op.replace_material=settings.replace_material
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        props = scene.syntexmexsettings

        ex_img = props.example_image

        #calculate algorithm properties and display some help how to set
        #it up correctly
        col2 = layout.column()
        if ex_img:
            (res_ex,res_ta,scaling,
             res_patch, res_patch0,
             overlap, overlap0, 
             mem_reqs, maxmem) = calc_syntexmex_info(ex_img, props)
            if props.synth_progress > 0.0001:
                showtext=f"synthesize texture...\npress 'ESC' to cancel."
            elif mem_reqs > maxmem:
                showtext=f"algorithm uses too much memory: \n~{mem_reqs:.2f} GB\nmore than the available {maxmem:.2f} GB.\nMaybe close down other \nprograms to free up memory?"
                canrun=False
                col2.alert=True
            elif np.any(np.array(res_patch)<2):
                showtext=f"algorithm needs patch\nsizes >2 to run!"
                col2.alert=True
                canrun=False
            elif np.any(np.array(overlap)<1):
                showtext=f"algorithm needs overlap\nsizes > 1"
                col2.alert=True
                canrun=False
            else:
                showtext="algorithm is ready..."
                canrun=True
        elif not ex_img:
            showtext="choose an example img.!"
            col2.alert=True
        elif props.synth_progress > 0.0001:
            showtext="press 'ESC' to stop texture synthesis"
        
        #display helptext
        col2.scale_y = 0.7
        multiline_label(col2,"> "+showtext)
        
        layout.separator(factor=2.0)
        if props.synth_progress < 0.0001: #if algorithm isnt running
            ######  algorithm start buttons for different run modes
            col3 = layout.column()
            if ex_img and canrun: col3.enabled=True
            else: col3.enabled=False
            col2 = col3.column()
            col2.scale_y = 2.0
            op1 = col2.operator("texture.syntexmex", 
                                text = "Synthesize UV example based texture")
            
            #col.scale_y = 1.0
            col3.separator(factor=2.0)
            op2 = col3.operator("texture.syntexmex", 
                                text = "Synthesize textures to UV islands")
            
            op3 = col3.operator("texture.syntexmex",
                                text = "Make UV seams seamless")
            
            if ex_img and canrun:
                self.copy_to_operator(op1,props)
                self.copy_to_operator(op2,props)
                op2.synth_tex=True
                op2.seamless_UVs=False
                self.copy_to_operator(op3,props)
                op3.synth_tex=False
                op3.seamless_UVs=True


            #####algorithm properties
            layout.separator(factor=2.0)             
            layout.prop(props, "replace_material")
            #for prop in scene.syntexmexsettings.__annotations__.keys():
            #layout.label(text="Example Material")
            layout.prop_search(props,"source_material", bpy.data, "materials",
                               text="Source Mat.")
            if props.source_material:
                layout.prop(props, "material_textures",text="Source Tex.")
            else:    
                layout.prop_search(props, "example_image", bpy.data, "images",
                               text="Source Tex.")
            #layout.prop_search(props,"example_image", props, "material_textures",
            #                    text="Source Tex.")
            layout.prop(props, "example_scaling", text='Ex. Scaling')
            layout.prop(props, "libsize")
            #TODO: make patch_size dependend on target_texture
            layout.prop(props, "patch_size")
            layout.prop(props, "target_resolution")
            layout.prop(props, "seed_value")


class syntexmex_info_panel(bpy.types.Panel):
    bl_label = "Info Panel"
    bl_idname = "SYNTEXMEX_PT_syntexmex_info_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'syntexmex'
    #bl_context = "tool"
    #bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.syntexmexsettings
        
        ex_img = props.example_image
        
        if ex_img:
            (res_ex,res_ta,scaling,
             res_patch, res_patch0,
             overlap, overlap0, 
             mem_reqs, maxmem) = calc_syntexmex_info(ex_img, props)            
            
            #layout.box()
            layout.prop(props, "synth_progress")
            layout.enabled=False
            
            layout.label(text="Algorithm Data:")
            b = layout.box()
            b.scale_y = 0.3

            multiline_label(b,f"""memory requirements: {mem_reqs:.2f}/{maxmem:.2f} GB
source scaling: {props.example_scaling*100:.0f}%
source resolution: [{res_ex[1]:.0f} {res_ex[0]:.0f}] px
target resolution: {res_ta[::-1]} px
synth resolution scaling: {scaling:.2f}
patches (highres): {res_patch0[::-1]} px
patches (lowres): {res_patch[::-1]} px
overlap (highres): {overlap0[::-1]}
overlap (lowres): {overlap[::-1]}""")  

class syntexmex_advanced_panel(bpy.types.Panel):
    bl_label = "Advanced Settings"
    bl_idname = "SYNTEXMEX_PT_syntexmex_advanced_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'syntexmex'
    #bl_context = "tool"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        col = layout.column()

        scene = context.scene
        props = scene.syntexmexsettings

        col.label(text="debugging options:")
        col.prop(props,"advanced_debugging", 
                 text="enable advanced console logs")


def update_debugging(self, context):
    scene = context.scene
    props = scene.syntexmexsettings

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Set advanced infos to: {props.advanced_debugging}")
    if props.advanced_debugging:
        logging.disable(logging.NOTSET)
    else:
        logging.disable(logging.INFO)
    return None



class synth_PBR_texture(bpy.types.Operator):
    """This operator synthesizes PBR textures from a synthmap
    (an image in which the pixel values are coordinates to an example image)"""
    bl_idname = "texture.syntexmex_pbr_texture"
    bl_label = "Synthesize PBR textures using a synth-map"
    bl_category = 'syntexmex'
    bl_options = {'REGISTER', 'UNDO'}
    
    synth_map: bpy.props.StringProperty()
    source_material : bpy.props.StringProperty()
    source_image : bpy.props.StringProperty()
    
    def execute(self, context):
        logger.info("start PBR synthesis")
        
        logger.info(f"synthesize textures: {self.synth_map},"\
                    f"{self.source_material},{self.source_image}")
        
        synthmap = up.blimage2array(bpy.data.images[self.synth_map])[...,:3]
    
        #scn = context.scene
        obj = context.active_object
    
        if self.source_material is not None:
            logger.info("create new seamless material")
            obj.active_material = mat = bpy.data.materials[self.source_material].copy()

            #duplicate texture and duplicate images inside
            #TODO: make "original" texture permanently saved
            imgnodes = [n for n in mat.node_tree.nodes if n.type=='TEX_IMAGE']
            images = [up.blimage2array(n.image)[...,:3] for n in imgnodes]
            synth_images = [us.reconstruct_synthmap(synthmap,
                                                    img, 
                                                    mode='normalized')
                            for img in images]
            
        elif self.source_image is not None:
            images = [up.blimage2array(self.source_image)[...,:3]]
            
        for simg,node in zip(synth_images,imgnodes):
            logger.info("processing image: {name}")
            if node.image.colorspace_settings.name=='Non-Color': is_data=True
            else: is_data=False
            new_img = bpy.data.images.new(node.image.name+"seamless",
                              synthmap.shape[1],synthmap.shape[0],
                              alpha=False,float_buffer=node.image.is_float,
                              is_data=is_data)
            #add alpha channel and upload into blender
            new_img.pixels[:] = np.dstack((simg,np.ones(simg.shape[:2]))).flatten()
            
            #new_img.colorspace_settings.name = node.image.colorspace_settings.name
            node.image = new_img
            new_img.update()
            new_img.pack()
            
        
        #us.reconstruct_synthmap(self.)
        return {'FINISHED'}



#TODO: if we want a list o something:
#https://sinestesia.co/blog/tutorials/using-uilists-in-blender/
class syntexmex_texture_operations_panel(bpy.types.Panel):
    bl_label = "texture operations"
    bl_idname = "SYNTEXMEX_PT_syntexmex_texture_operations_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'syntexmex'
    #bl_context = "tool"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False
        ob = context.active_object

        scene = context.scene
        props = scene.syntexmexsettings
        op = layout.operator("texture.syntexmex_pbr_texture", 
                           text = "Synthesize UV example based texture")
        
        ex_img = props.example_image
        ta_img = props.target_image
        img_init=(ex_img is not None) and (ta_img is not None)
        
        if props.active_synthmap:
            op.synth_map = props.active_synthmap.name
        if props.source_material:
            op.source_material = props.source_material.name
        if props.source_image:
            op.source_image = props.source_image.name

        layout.use_property_split = True
        layout.label(text="active synthmap:")
        layout.template_ID(props,"active_synthmap", open="image.open")
        layout.prop_search(props,"source_material", bpy.data, "materials")
        layout.label(text="source image:")
        layout.template_ID(props,"source_image", open="image.open")
        
        
#    if prop in images:
        layout.label(text="Example Image:")
        layout.template_ID(props, "example_image", open="image.open")
        layout.prop(props, "example_scaling")
        layout.prop(props, "libsize")
        #TODO: make patch_size dependend on target_texture
        layout.prop(props, "patch_size")


        layout.separator(factor=2.0)
        layout.label(text="Target Image:")
        layout.template_ID(props, "target_image", 
                        new="image.new",open="image.open")
        if (ta_img is not None):
            layout.operator("texture.clear_target_texture")

def get_textures_from_material(self, context):
    mat = self.source_material
    texlist = []
    if mat is not None:
        for i,n in enumerate(mat.node_tree.nodes):
            if n.type=="TEX_IMAGE":
                texlist.append((n.image.name,n.image.name,"",i))
                #print(f"updating images!{n.image.name},{i}")
    return texlist
    #images = [n.image for n in mat.node_tree.nodes if n.type=='TEX_IMAGE']
    #return None#images

def update_source_image_from_enum(self, context):
    props = context.scene.syntexmexsettings
    sel_tex = props.material_textures
    props.example_image = bpy.data.images[sel_tex]
    
class SyntexmexSettings(bpy.types.PropertyGroup):
    #https://docs.blender.org/api/current/bpy.props.html
    synth_progress: bpy.props.FloatProperty(
        name="synthetization progress",
        description="synthetization progress",
        min=0.0,max=1.0,
        subtype = "PERCENTAGE"
        )
    patch_size: bpy.props.FloatProperty(
        name="Patch Size Ratio",
        description="Set width of patches as a ratio of shortest edge of an image",
        min=0.0001,
        max=0.5,
        default=0.1,
        precision=3,
        step=0.1,
        )
    example_image: bpy.props.PointerProperty(name="Ex.Tex.", type=bpy.types.Image)
    example_scaling: bpy.props.FloatProperty(
        name="Example Scaling",
        description="""Scale Example to a certain size which will be used
to generate the texture""",
        min=0.0001,
        max=1.0,
        default=1.0,
        precision=3,
        step=1
    )
    target_image: bpy.props.PointerProperty(name="target tex", type=bpy.types.Image)
    target_resolution: bpy.props.IntVectorProperty(name="synth-res.",
                                                   size=2,
                                                   min=0,
                                                   max=10000,
                                                   default=(1024,1024))
    libsize: bpy.props.IntProperty(name="Library Size",
            description="defines the quality of the texture (higher=better, but needs more memory)",
            min=10*10, default = 128*128,
            step=10#TODO: currently not implemented in blender
            )
    seed_value: bpy.props.IntProperty(
            name="Seed Value",
            description="Seed value for predictable texture generation",
            default = 0,
            )
    source_material : bpy.props.PointerProperty(name="source material", 
                                                type=bpy.types.Material)
    #material_textures : bpy.props.CollectionProperty(type=TextureList)
    #material_textures : bpy.props.CollectionProperty(type=bpy.types.Image)
    material_textures : bpy.props.EnumProperty(items = get_textures_from_material,
                                               update = update_source_image_from_enum)
    source_image : bpy.props.PointerProperty(name="source image",
                                             type=bpy.types.Image)
    active_synthmap : bpy.props.PointerProperty(name="active synthmap", 
                                                  type=bpy.types.Image)
    advanced_debugging: bpy.props.BoolProperty(
            name="Advanced Debugging",
            description="Enable Advanced Debugging (in console)",
            default = True,
            update = update_debugging
            )
    replace_material : bpy.props.BoolProperty(
            name="repplace material",
            description=("replace all textures in material with"
                        "a seamless version"),
            default=True
            )


classes = (
    syntexmex_info_panel,
    syntexmex_panel,
    syntexmex_texture_operations_panel,
    syntexmex_advanced_panel,
    syntexmex,
    synth_PBR_texture,
    clear_target_texture
)
register_panel, unregister_panel = bpy.utils.register_classes_factory(classes)


def register():
    #syntexmex_panel.register()
    bpy.utils.register_class(SyntexmexSettings)
    bpy.types.Scene.syntexmexsettings = bpy.props.PointerProperty(type=SyntexmexSettings)
    register_panel()
    #init propertygroup

def unregister():
    #syntexmex_panel.unregister()
    unregister_panel()
    bpy.utils.unregister_class(SyntexmexSettings)
    del(bpy.types.Scene.syntexmexsettings)
    #del(bpy.context.scene['syntexmexsettings'])

if __name__ == "__main__":
    logger.info("register syntexmex")
    # pipe_operator.register()
    # bpy.ops.mesh.add_pipes()
    register()
    # debugging
    #bpy.ops.mesh.add_pipes(number=5, mode='skin', seed = 11)

#obj = bpy.context.selected_objects[0]
