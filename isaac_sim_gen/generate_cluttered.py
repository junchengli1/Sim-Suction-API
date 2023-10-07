import argparse
from pathlib import Path
base_dir = Path(__file__).parent
parser = argparse.ArgumentParser(description='Dataset_generation')
parser.add_argument('--headless', type=bool, default=False, help='headless')
parser.add_argument('--asset_path', type=str, default=(base_dir.parent / 'nvidia_assets/*.usd').as_posix(), help='nvidia assets')
parser.add_argument('--asset_path1', type=str, default=(base_dir.parent / 'ShapeNet/*.usd').as_posix(), help='shapenet (subset)')
parser.add_argument('--save_data_path', type=str, default=(base_dir.parent / "synthetic_data").as_posix(), help='data path')
parser.add_argument('--objects_per_stage', type=int, nargs=2, default=[1,20], help='Minimum and maximum number of objects per stage')
parser.add_argument('--max_stage', type=int,default=500, help='number of cluttered environments')

args = parser.parse_args()

from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless": args.headless})
from omni.isaac.core import World
from pxr import Gf,UsdGeom, UsdPhysics
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni
import carb
import numpy as np
import os
import glob
import pickle
import json 
import random
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.prims import XFormPrim
from omni.isaac.dynamic_control import _dynamic_control
import typing
import copy
import matplotlib.pyplot as plt
from typing import Optional
from pxr import Usd, UsdGeom, Gf
from omni.physx.scripts import utils
import asyncio
import scipy.stats as stats


ENABLE_PHYSICS = True
simulation_world = World(stage_units_in_meters = 0.01)
# Step our simulation to ensure everything initialized
simulation_world.clear()
simulation_world.step()

# SCENE SETUP
set_camera_view(eye=np.array([0, 0, 500]), target=np.array([0, 0, 0]))

async def pause_sim(task):
    done, pending = await asyncio.wait({task})
    if task in done:
        print("Waited until next frame, pausing")
        omni.timeline.get_timeline_interface().pause()

def get_world_transform_xform(prim: Usd.Prim) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using Xformable.
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale
    
def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_imageable.html

    Args:
        prim: A prim to compute the bounding box.
    Returns:
        A range (i.e. bounding box), see more at: https://graphics.pixar.com/usd/release/api/class_gf_range3d.html
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return bound_range
    
class BaseTask(object):
    """This class provides a way to set up a task in a scene and modularize adding objects to stage,
        getting observations needed for the behavioral layer, caclulating metrics needed about the task,
        calling certain things pre-stepping, creating multiple tasks at the same time and much more.
    
        Checkout the required tutorials at 
        https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html

        Args:
            name (str): needs to be unique if added to the World.
            ffset (Optional[np.ndarray], optional): offset applied to all assets of the task.
        """

    def __init__(self, name: str, offset: Optional[np.ndarray] = None) -> None:
        self._scene = None
        self._name = name
        self._offset = offset
        self._task_objects = dict()
        if self._offset is None:
            self._offset = np.array([0.0, 0.0, 0.0])
        return

    @property
    def scene(self) -> Scene:
        """Scene of the world

        Returns:
            Scene: [description]
        """
        return self._scene

    @property
    def name(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return self._name

    def set_up_scene(self, scene: Scene) -> None:
        """Adding assets to the stage as well as adding the encapsulated objects such as XFormPrim..etc
           to the task_objects happens here.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        return


    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """Returns current observations from the objects needed for the behavioral layer.

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Returns True of the task is done.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        """Calls while doing a .reset() on the world.
        """
        return

    def get_description(self) -> str:
        """[summary]

        Returns:
            str: [description]
        """
        return ""

    def cleanup(self) -> None:
        """Called before calling a reset() on the world to removed temporarly objects that were added during
           simulation for instance.
        """
        return




        
class ObjectFreeDrop(BaseTask):

    def __init__(self, name: str = "bin_filling") -> None:
        BaseTask.__init__(self, name=name, offset=None)
        self._packing_bin = None

        ########################################################################################################33
        self._objects_asset_paths=glob.glob(args.asset_path)
        self._objects_asset_paths1=glob.glob(args.asset_path1)
        #self._bin=args.tote_path
        ################################################################################################################

        self._objects = []
        self._max_objects = 100
        self._objects_to_add = 0
        self._mesh_objects={}
        self._stage_count = 0
        return

    def get_current_num_of_objects_to_add(self) -> int:
        """
        Returns:
            int: Number of objects left to drop from the pipe
        """
        
        return self._objects_to_add

    def set_up_scene(self, scene: Scene) -> None:
        """Loads the stage USD and adds the packing bin to the World's scene.

        Args:
            scene (Scene): The world's scene.
        """
        super().set_up_scene(scene)

        #add_reference_to_stage(usd_path=self._bin, prim_path="/World/Scene/bin")

        return

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Executed before the physics step.

        Args:
            time_step_index (int): Current time step index
            simulation_time (float): Current simulation time.
        """
        BaseTask.pre_step(self, time_step_index=time_step_index, simulation_time=simulation_time)
        #self._ur10_robot.gripper.update()
        if self._objects_to_add > 0 and len(self._objects) < self._max_objects and time_step_index % 30 == 0:
            self._add_object()
        return

    def post_reset(self) -> None:
        """Executed after reseting the scene
        """
        self._objects_to_add = 0
        self._objects = []
        return

    def add_objects(self, objects_number: int = 10) -> None:
        """Adds number of objects to be added by the pipe

        Args:
            objects_number (int, optional): number of objects to be added by the pipe. Defaults to 10.
        """
        self._objects_to_add += objects_number
        return

    def _add_object(self):
        ###########total asset list################
        all_asset_paths = [self._objects_asset_paths, self._objects_asset_paths1]
        remaining_percentage = self._objects_to_add / (self._objects_to_add+ len(self._objects))  
        ##Update weights based on the remaining items to be dropped###########
        if remaining_percentage > 0.75:  # In the top 1/3 of items
            weights = [1, 1]
        elif remaining_percentage > 0.5:  # In the middle 1/2 of items
            weights = [2, 1]
        else:  # In the bottom 1/3 of items
            weights = [1, 2]

        # Sample asset path based on updated weights
        chosen_directory = random.choices(all_asset_paths, weights=weights, k=1)[0]
        asset_path = random.choice(chosen_directory)

        ##################rescale if needed########################
        if asset_path in self._objects_asset_paths:
            scale=np.array([1,1,1])   
        else:
            num=random.randint(10,25) #randomly choose a scale between 10 and 25 for shapenet objects
            scale=np.array([num,num,num])
        ##################rescale if needed########################

        _,object_name=os.path.split(asset_path)

        prim_path = "/World/objects/object_{}".format(len(self._objects))
        self._mesh_objects["/World/objects/object_{}".format(len(self._objects))]=dict()
        self._mesh_objects["/World/objects/object_{}".format(len(self._objects))]["object_name"]=asset_path
        
        current_orientation = R.from_euler('xyz', [0, 0, 0], degrees=True)

        # Define a small range for changing the orientation
        delta_range = 5  # degrees

        # Generate random changes in orientation within the delta range
        delta_orientation = R.from_euler('xyz', np.random.uniform(-delta_range, delta_range, size=3), degrees=True)

        # Apply the changes to the current orientation
        new_orientation = current_orientation * delta_orientation
        new_orientation_euler = new_orientation.as_euler('xyz', degrees=True)

        
        asset=create_prim(
                    prim_path=prim_path, 
                    scale = scale,
                    usd_path=asset_path,
                    orientation=euler_angles_to_quat(new_orientation_euler),
                    semantic_label=object_name
                )
     
        self._objects.append(prim_path)
        

        bound = UsdGeom.Mesh(asset).ComputeWorldBound(0.0, "default")
        box_min_y = bound.GetBox().GetMin()[1]

        xform_prim = XFormPrim(asset.GetPath())
        RANDOM_TRANSLATION_X = (-20, 20.0)
        RANDOM_TRANSLATION_y = (-5, 5.0)
        RANDOM_TRANSLATION_Z = (40.0, 80)
        x = random.uniform(*RANDOM_TRANSLATION_X)
        y = random.uniform(*RANDOM_TRANSLATION_y)
        z = random.uniform(*RANDOM_TRANSLATION_Z)
        xform_prim.set_world_pose(position =np.array([x,y,z]))  
        
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(asset)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        collisionAPI = UsdPhysics.CollisionAPI.Apply(asset)

        stage = omni.usd.get_context().get_stage()
     

        self._objects_to_add -= 1

        if  self._objects_to_add==0:

            path=args.save_data_path+f"/stage_{self._stage_count}" #self._stage_count is stage number
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            
            simulation_world.play()

            for i in range(1000):#give some time for the objects to drop and settle
                simulation_world.step(render=True)

            stage = omni.usd.get_context().get_stage()
            objects_to_remove = []
            simulation_world.pause()

            ############remove items that are not in perception range#####################
            for i in range(len(self._objects)):
                asset_current=stage.GetPrimAtPath("/World/objects/object_{}".format(i))
                bound = UsdGeom.Mesh(asset_current).ComputeWorldBound(0.0, "default")
                box_min_y = bound.GetBox().GetMin()[1]
                box_max_y = bound.GetBox().GetMax()[1]
                box_min_x = bound.GetBox().GetMin()[0]
                box_max_x = bound.GetBox().GetMax()[0]
                block_min =  [box_min_x, box_min_y, 0]
                block_max =  [box_max_x, box_max_y, 0]
                compute_bbox=get_world_transform_xform(asset_current)
                trans_item = list(compute_bbox[0])

                if trans_item[2] <= 40:

                    if (
                        not (-40 <= block_min[0] <= 40 and -40 <= block_min[1] <=40)  # if min coordinates are not within perception range
                        or
                        not (-40 <= block_max[0] <= 40 and -40 <= block_max[1] <= 40)  # if max coordinates are not within perception range
                        ):
                        objects_to_remove.append(self._objects[i])
                else:
                    objects_to_remove.append(self._objects[i])
            
            for rm_object_prim in set(objects_to_remove):
                stage.RemovePrim(rm_object_prim)
            
             ############remove items that are not in perception range#####################
          
            omni.usd.get_context().save_as_stage(path+f"/stage_{self._stage_count}"+".usd", None) #save stage

            self._stage_count+=1
            simulation_world.stop()
            if self._stage_count==args.max_stage:
               simulation_app.close()
        return

    def cleanup(self) -> None:
        """Removed the added objects when resetting.
        """
        stage = omni.usd.get_context().get_stage()
        for i in range(len(self._objects)):
            stage.RemovePrim(self._objects[i])

        self._objects = []
        return



if __name__ == "__main__":
    if ENABLE_PHYSICS:
        # Create a ground plane
        simulation_world.scene.add_ground_plane(size=1000, color=np.array([1, 1, 1]))

    my_task = ObjectFreeDrop()
    simulation_world.add_task(my_task)
    simulation_world.reset()
    simulation_world.step()

    added_objects_flag = False
    while simulation_app.is_running():
        if simulation_world.is_playing():
            simulation_world.step(render=True)
            if simulation_world.current_time_step_index == 0:
                simulation_world.reset()
                added_objects_flag = False
            if not added_objects_flag:
                
                ##################choose objects total number to drop####################
                min_object_number=args.objects_per_stage[0]
                max_object_number=args.objects_per_stage[1]
                values = np.arange(min_object_number, max_object_number)  # Values from min_object_number to max_object_number
                # Define the probabilities
                
                ##################Detailed control of the object number####################

                # Generate a normal distribution centered around the mean value
                mean =  (min_object_number + max_object_number) / 2
                std_dev = 5  # You can adjust this for a narrower or wider spread
                probs = stats.norm.pdf(values, mean, std_dev)

                # Make sure probabilities sum to 1
                probs /= probs.sum()
                #print(probs,probs.sum())
                chosen_number = np.random.choice(values, p=probs)
                print(chosen_number)
                my_task.add_objects(objects_number=chosen_number)
                added_objects_flag = True
            
        else:
            simulation_world.render()
    simulation_app.close()
