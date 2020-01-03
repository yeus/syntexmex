def main(context):
    obj = context.active_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    uv_layer = bm.loops.layers.uv.verify()

    # adjust uv coordinates
    for face in bm.faces:
        for loop in face.loops:
            loop_uv = loop[uv_layer]
            # use xy position of the vertex as a uv coordinate
            loop_uv.uv = loop.vert.co.xy

    bmesh.update_edit_mesh(me)


class TextureUnwrapper(bpy.types.Operator):
    """This operator takes an image"""
    bl_idname = "uv.texture_unwrapper"
    bl_label = "Simple UV Operator"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        main(context)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(UvOperator)


def unregister():
    bpy.utils.unregister_class(UvOperator)

if __name__ == "__main__":
    #register()

    # test call
    #bpy.ops.uv.simple_operator()
    
    #bpy.ops.image.open(filepath="//rpitex.png", directory="/home/tom/Dropbox/company/stythreedee/", files=[{"name":"rpitex.png", "name":"rpitex.png"}], relative_path=True, show_multiview=False)

    #TODO: vielleich auch Flächengetreu?
    #bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    
    #new_tex = bpy.data.images.new("textureunwrap", width=128, height=128)
    pass 
