from omni.isaac.kit import SimulationApp
CONFIG = {"headless":True}
simulation_app=SimulationApp(launch_config=CONFIG)
import omni.usd
import omni.client

from pxr import UsdGeom, Sdf
import os 
import glob
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim,delete_prim
from omni.physx.scripts import utils
from pxr import Gf, Sdf, UsdGeom, UsdShade, Semantics, UsdPhysics
from pxr import UsdGeom, Usd

from pathlib import Path
import argparse
base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='convert instanceable')
parser.add_argument('--headless', type=bool, default=False, help='headless')#make it True when gerneate dataset 
parser.add_argument('--data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')



def hasSchema(prim, schemaName):
    schemas = prim.GetAppliedSchemas()
    for s in schemas:
        if s == schemaName:
            return True
    return False

def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):

    """ Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
        Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

        Args:
            asset_usd_path (str): USD file path for asset
            source_prim_path (str): USD path of root prim
            save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)

def convert_asset_instanceable(asset_usd_path, source_prim_path, save_as_path=None, create_xforms=False):

    """ Makes all mesh/geometry prims instanceable.
        Can optionally add UsdGeom.Xform prim as parent for all mesh/geometry prims.
        Makes a copy of the asset USD file, which will be used for referencing.
        Updates asset file to convert all parent prims of mesh/geometry prims to reference cloned USD file.

        Args:
            asset_usd_path (str): USD file path for asset
            source_prim_path (str): USD path of root prim
            save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
            create_xforms (bool): Whether to add new UsdGeom.Xform prims to mesh/geometry prims.
    """

    if create_xforms:
        create_parent_xforms(asset_usd_path, source_prim_path, save_as_path)
        asset_usd_path = save_as_path

    instance_usd_path = ".".join(asset_usd_path.split(".")[:-1]) + "_meshes.usd"
    omni.client.copy(asset_usd_path, instance_usd_path)
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(root_prim)
    delete_prim("/World/groundPlane")

    prims = [stage.GetPrimAtPath(source_prim_path)]
    curr_prim = [stage.GetPrimAtPath("/World")]
    
    obj_prim = stage.GetPrimAtPath("/World/objects")

    for prim in Usd.PrimRange(obj_prim):
            if (prim.IsA(UsdGeom.Xform)):
                if hasSchema(prim, "PhysicsRigidBodyAPI"):
                   pass
                else:
                    utils.setRigidBody(prim, "convexDecomposition", True)
     
                mass_api=UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateDensityAttr(0.0001)
      

    for prim in Usd.PrimRange(obj_prim):
            if (prim.IsA(UsdGeom.Mesh)):
                utils.setCollider(prim, approximationShape="none")


    if len(curr_prim)!=0:
        prims = curr_prim

    while len(prims) > 0:
        prim = prims.pop(0)
        if prim:
            if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                parent_prim = prim.GetParent()
                if parent_prim and not parent_prim.IsInstance():
                    parent_prim.GetReferences().AddReference(assetPath=instance_usd_path, primPath=str(parent_prim.GetPath()))
                    parent_prim.SetInstanceable(True)
                    continue

            children_prims = prim.GetChildren()
            prims = prims + children_prims

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)


if __name__ == "__main__":
    # Run the main function
    args = parser.parse_args()
    for stage_ind in range(0,500):
        asset_usd_path=args.data_path+f"/stage_{stage_ind}"+ f"/stage_{stage_ind}.usd"
        save_as_path=args.data_path+f"/stage_{stage_ind}"+f"/stage_{stage_ind}_instanceable.usd"
        source_prim_path="/World"
        convert_asset_instanceable(asset_usd_path,source_prim_path,save_as_path)
      

