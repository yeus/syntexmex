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

#
#blend_dir = os.path.dirname(bpy.data.filepath)
# if blend_dir not in sys.path:
#   sys.path.append(blend_dir)
# temporarily appends the folder containing this file into sys.path
main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.append(main_dir)

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


class synthesize_uv_islands(bpy.types.Operator):
    """Synthesizes textures to UV islands"""
    bl_idname = "texture.synthesize_uv_islands"
    bl_label = "Synthesize UV islands"

    def execute(self, context):
        logger.info("start synthesizing UV islands")
        return {'FINISHED'}

class make_seamless(bpy.types.Operator):
    """Synthesizes textures to UV islands"""
    bl_idname = "texture.make_seamless"
    bl_label = "make UV seams seamless"

    def execute(self, context):
        logger.info("make UV seams seamless")
        return {'FINISHED'}

class TextureUnwrapper(bpy.types.Operator):
    """This operator takes an image"""
    bl_idname = "texture.texture_unwrapper"
    bl_label = "Simple UV Operator"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        main(context)
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

        layout.operator("texture.synthesize_uv_islands")
        layout.operator("texture.make_seamless")
        #layout.operator("object.piperator_delete")


classes = (
    syntexmex_panel,
    synthesize_uv_islands,
    make_seamless
)
register_panel, unregister_panel = bpy.utils.register_classes_factory(classes)


def register():
    #syntexmex_panel.register()
    register_panel()


def unregister():
    #syntexmex_panel.unregister()
    unregister_panel()


if __name__ == "__main__":
    # pipe_operator.register()
    # bpy.ops.mesh.add_pipes()
    register()

    # debugging
    #bpy.ops.mesh.add_pipes(number=5, mode='skin', seed = 11)

#obj = bpy.context.selected_objects[0]
