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
main_dir = os.path.dirname(bpy.data.filepath)
sys.path.append(main_dir)
main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.append(main_dir)
print(main_dir)

import textureunwrap as tu
importlib.reload(tu)

#TODO: from . import pipe_operator
#importlib.reload(pipe_operator)


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
    target_tex: bpy.props.StringProperty()
    synth_tex: bpy.props.BoolProperty(name="synthesize textures")
    seamless_UVs: bpy.props.BoolProperty(name="seamless UV islands")

    @classmethod
    def poll(cls, context):
        #TODO: this operator should also be able to execute in other
        # windows as well
        return context.space_data.type in {'VIEW_3D'}
    
    def execute(self, context):
        logger.info("start synthesizing algorithm")
        scene = context.scene
        
        #TODO: select specific object in menu
        obj, bm = tu.create_bmesh_from_active_object()
        uv_layer = bm.loops.layers.uv['UVMap']
        
        print(self.example_image)
        print(self.target_tex)
        
        example_image = bpy.data.images[self.example_image]
        target_tex = bpy.data.images[self.target_tex]
        
        examplebuf, targetbuf = tu.init_texture_buffers(example_image,
                                    target_tex, self.example_scaling)
        target = tu.synthesize_textures_algorithm(examplebuf, 
                                        targetbuf,
                                        bm,
                                        synth_tex=self.synth_tex,
                                        seamless_UVs=self.seamless_UVs)
        
        # Write back to blender image.
        target_tex.pixels[:] = target.flatten()
        target_tex.update()
        #nt.save()
        logger.info("finished synthesizing!")
        return {'FINISHED'}
    
class clear_target_texture(bpy.types.Operator):
    """Clear target texture to a generic texture"""
    bl_idname = "texture.clear_target_texture"
    bl_label = "Clear target texture"

    def execute(self, context):
        logger.info("make UV seams seamless")
        return {'FINISHED'}


#TODO: ame everything according to here:
#https://wiki.blender.org/wiki/Reference/Release_Notes/2.80/Python_API/Addons
class syntexmex_panel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Syntexmex configuration panel"
    bl_idname = "SYNTEXMEX_PT_syntexmex_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'syntexmex'
    #bl_context = "tool"

    def draw(self, context):
        layout = self.layout

        scene = context.scene

        #layout.box()
        col = layout.column()
        op1 = col.operator("texture.syntexmex", 
                            text = "Synthesize UV example based texture")  
        op1.example_image = scene.syntexmexsettings.example_image.name
        op1.target_tex = scene.syntexmexsettings.target_tex.name 
        op1.example_scaling=scene.syntexmexsettings.example_scaling     
        
        col.separator(factor=2.0)
        op2 = col.operator("texture.syntexmex", 
                            text = "Synthesize textures to UV islands")
        op2.example_image = scene.syntexmexsettings.example_image.name
        op2.target_tex = scene.syntexmexsettings.target_tex.name
        op2.synth_tex=True
        op2.example_scaling=scene.syntexmexsettings.example_scaling
        op3 = col.operator("texture.syntexmex",
                            text = "Make UV seams seamless")
        op3.example_image = scene.syntexmexsettings.example_image.name
        op3.target_tex = scene.syntexmexsettings.target_tex.name
        op3.example_scaling=scene.syntexmexsettings.example_scaling
        op3.seamless_UVs=True
        
        col.separator(factor=2.0)
        col.operator("texture.clear_target_texture")
        col.separator(factor=2.0)
        #taken from here: https://blender.stackexchange.com/questions/72402/how-to-iterate-through-a-propertygroup
        b = col.box()
        b.label(text="Algorithm Data:")
        b.label(text="TODO")
        col.separator(factor=2.0)
        images = ["example_image","target_tex"]
        for prop in scene.syntexmexsettings.__annotations__.keys():
            if prop in images:
                col.label(text=prop)
                col.template_ID(scene.syntexmexsettings, prop, 
                                new="image.new",open="image.open")
            else:
                col.prop(scene.syntexmexsettings, prop)
        #col.prop(scene.syntexmexsettings,None)
        #TODO: make it possible to open images
        #
        #layout.operator("object.piperator_delete")

class SyntexmexSettings(bpy.types.PropertyGroup):
    #https://docs.blender.org/api/current/bpy.props.html
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
    target_tex: bpy.props.PointerProperty(name="target tex", type=bpy.types.Image)



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

def unregister():
    #syntexmex_panel.unregister()
    unregister_panel()
    bpy.utils.unregister_class(SyntexmexSettings)
    del(bpy.types.Scene.syntexmexsettings)

if __name__ == "__main__":
    logger.info("register syntexmex")
    # pipe_operator.register()
    # bpy.ops.mesh.add_pipes()
    register()

    # debugging
    #bpy.ops.mesh.add_pipes(number=5, mode='skin', seed = 11)

#obj = bpy.context.selected_objects[0]
