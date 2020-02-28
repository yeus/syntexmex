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
main_dir = os.path.dirname(bpy.data.filepath) #blender directory
sys.path.append(main_dir)
main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib') #addon dir "__init__.py" + lib
sys.path.append(main_dir)
print(main_dir)

import uv_prepare as up
importlib.reload(up)
import uv_synthesize as us
importlib.reload(us)


__author__ = "yeus <Thomas.Meschede@web.de>"
__status__ = "test"
__version__ = "0.9"
__date__ = "2020 Feb 29th"

#TODO: fill this out
bl_info = {
    "name": "TODO",
    "author": "yeus <Thomas.Meschede@web.de",
    "version": (0, 0, 91),
    "blender": (2, 82, 0),
    "location": "TODO",
    "description": "TODO",
    "warning": "",
    "wiki_url": "http://yeus.gitlab.io",
    "category": "UV",
    "support": "TESTING",
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
    synth_tex: bpy.props.BoolProperty(name="synthesize textures")
    seamless_UVs: bpy.props.BoolProperty(name="seamless UV islands")
    libsize: bpy.props.IntProperty(name="patch library size")

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
        
        print(self.example_image)
        print(self.target_image)
        
        example_image = bpy.data.images[self.example_image]
        self.target = bpy.data.images[self.target_image]
        
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
      
    def write_image(self,target):
        # Write back to blender image.
        self.target.pixels[:] = target.flatten()
        self.target.update()
        #nt.save()
        logger.info("synced images!")
        #return {'FINISHED'}
        #return {'RUNNING_MODAL'}"""

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
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            logger.info("thread was cancelled!")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # change theme color, silly!
            if self._thread.is_alive():
                target = self.receive(context)
                #print(".", end = '')
            else:
                logger.info("thread seems to have finished!")
                self.receive(context)
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
            self.write_image(msg)
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
        
        return {'FINISHED'}

class uvs2json(bpy.types.Operator):
    """Save UVs as json datafile"""
    bl_idname = "texture.uvs2json"
    bl_label = "Save UVs to json file"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        logger.info("save UVs")
        ta = context.scene.syntexmexsettings.target_image
        
        #target = np.zeros((ta.size[1], ta.size[0],4))
        #target[...,3]=1.0
        
        ta.pixels[:] = target.flatten()
        ta.update()
        
        return {'FINISHED'}

#TODO: ame everything according to here:
#https://wiki.blender.org/wiki/Reference/Release_Na.otes/2.80/Python_API/Addons
class syntexmex_panel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Syntexmex configuration panel"
    bl_idname = "SYNTEXMEX_PT_syntexmex_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'syntexmex'
    #bl_context = "tool"

    def copy_to_operator(self, op, settings):
        op.example_image = settings.example_image.name
        op.target_image = settings.target_image.name 
        op.example_scaling = settings.example_scaling  
        op.libsize = settings.libsize 
        op.patch_size = settings.patch_size 
        op.synth_tex=True      
        op.seamless_UVs=True
    
    def draw(self, context):
        layout = self.layout

        scene = context.scene

        #layout.box()
        col = layout.column()
        col2 = col.column()
        col2.prop(scene.syntexmexsettings, "synth_progress")
        col2.enabled=False
        col.separator(factor=2.0)
        if scene.syntexmexsettings.synth_progress < 0.0001: #if algorithm isn running
            props = scene.syntexmexsettings
            ex_img = props.example_image
            ta_img = props.target_image
            if (ex_img is not None) and (ta_img is not None):
                col2 = col.column()
                col2.scale_y = 2.0
                op1 = col2.operator("texture.syntexmex", 
                                    text = "Synthesize UV example based texture")
                self.copy_to_operator(op1,scene.syntexmexsettings)
                
                #col.scale_y = 1.0
                col.separator(factor=2.0)
                op2 = col.operator("texture.syntexmex", 
                                    text = "Synthesize textures to UV islands")
                self.copy_to_operator(op2,scene.syntexmexsettings)
                op2.synth_tex=True
                op2.seamless_UVs=False
                op3 = col.operator("texture.syntexmex",
                                    text = "Make UV seams seamless")
                self.copy_to_operator(op3,scene.syntexmexsettings)
                op3.synth_tex=False
                op3.seamless_UVs=True
                
                col.separator(factor=2.0)
                #taken from here: https://blender.stackexchange.com/questions/72402/how-to-iterate-through-a-propertygroup
                col.label(text="Algorithm Data:")
                b = col.box()
                b.scale_y = 0.3

                res_ex = np.array((ex_img.size[1],ex_img.size[0])) * props.example_scaling
                res_ta = np.array((ta_img.size[1],ta_img.size[0]))
                scaling = us.ts.calc_lib_scaling(res_ex, props.libsize)
                (res_patch, res_patch0,
                overlap, overlap0) = us.ts.create_patch_params2(res_ex,
                                                         scaling,
                                                         1/6.0, props.patch_size)
            #mem_reqs = tu.ts.check_memory_requirements2(res_ex,
            #                        res_patch, ch_num, )
                mem_reqs = us.ts.check_memory_requirements2(res_ex*scaling,
                                          res_patch,
                                          ch_num = 3, 
                                          itemsize = 8)
                multiline_label(b,f"""memory requirements: {mem_reqs:.2f} GB
source scaling: {props.example_scaling*100:.0f}%
source resolution: [{res_ex[1]:.0f} {res_ex[0]:.0f}] px
target resolution: {res_ta[::-1]} px
synth resolution scaling: {scaling:.2f}
patches (highres): {res_patch0[::-1]} px
patches (lowres): {res_patch[::-1]} px
overlap (highres): {overlap0[::-1]}
overlap (lowres): {overlap[::-1]}""")
            else:
                b = col.box()
                b.scale_y = 0.3
                multiline_label(b,">>> please choose a target &\nexample image")
                
            col.separator(factor=2.0)
            #for prop in scene.syntexmexsettings.__annotations__.keys():
            #    if prop in images:
            col.label(text="Example Image:")
            col.template_ID(scene.syntexmexsettings, "example_image", 
                            new="image.new",open="image.open")

            col.label(text="Target Image:")
            col.template_ID(scene.syntexmexsettings, "target_image", 
                            new="image.new",open="image.open")
            if (ta_img is not None):
                col.operator("texture.clear_target_texture")

            col.separator(factor=2.0)
            col.prop(scene.syntexmexsettings, "patch_size")
            col.prop(scene.syntexmexsettings, "example_scaling")
            col.prop(scene.syntexmexsettings, "libsize")
            #col.prop(scene.syntexmexsettings,None)
            #TODO: make it possible to open images
            #
            #layout.operator("object.piperator_delete")

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
        min=0.0,
        max=0.5,
        default=0.1,
        precision=3,
        step=0.1
        )
    example_image: bpy.props.PointerProperty(name="Ex.Tex.", type=bpy.types.Image)
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
    target_image: bpy.props.PointerProperty(name="target tex", type=bpy.types.Image)
    libsize: bpy.props.IntProperty(name="library size",
                        description="defines the quality of the texture (higher=better, but needs more memory)", min=0, default = 128*128)



classes = (
    syntexmex_panel,
    syntexmex,
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
