# NONFUNCTIONAL 

import math
import pathlib
import Sofa  # SofaPython3

# ===================== User settings =====================
MESH_DIR   = pathlib.Path(__file__).resolve().parent
HEMI_MESH  = MESH_DIR / "hemisphere.msh"
IND_STL    = MESH_DIR / "indenter.stl"

DT         = 0.002         # time step [s]
GRAVITY    = [0.0, 0.0, 0.0]

# Soft material params (example)
YOUNG      = 1.0e6         # Pa
POISSON    = 0.45
DENSITY    = 1000.0        # kg/m^3 (quasi-static; not critical but required)

# Contact detection distances (set ~ to your surface element size)
ALARM_DIST   = 2.0e-3      # [m]
CONTACT_DIST = 1.0e-3      # [m]
FRICTION     = 0.0         # start frictionless

# Motion: down–hold–up
Z_TRAVEL   = -0.010        # [m] (negative is downward)
T_DOWN     = 2.0           # [s]
T_HOLD     = 2.0           # [s]
T_UP       = 2.0           # [s]
T_TOTAL    = T_DOWN + T_HOLD + T_UP

# ===================== Motion profile =====================
def z_of_t(t: float) -> float:
    """Piecewise linear: down, hold, up; returns displacement (m)."""
    if t <= T_DOWN:
        return (t / T_DOWN) * Z_TRAVEL
    elif t <= T_DOWN + T_HOLD:
        return Z_TRAVEL
    elif t <= T_TOTAL:
        tau = (t - (T_DOWN + T_HOLD)) / T_UP
        return Z_TRAVEL * (1.0 - tau)
    else:
        return 0.0

# ===================== Controller =====================
class IndentationController(Sofa.Core.Controller):
    """Prescribes rigid indenter z-motion and logs forces."""
    def __init__(self, node=None, rigidMO=None, monitor=None):
        super().__init__()
        self.node = node
        self.mo = rigidMO
        self.mon = monitor
        self.t = 0.0
        self.q0 = None

    def onLoaded(self):
        # Store initial pose [x y z qx qy qz qw]
        self.q0 = list(self.mo.position.value[0])
        return 0

    def onBeginAnimationStep(self, dt):
        self.t += dt
        q = self.q0.copy()
        q[2] = self.q0[2] + z_of_t(self.t)  # move along z
        self.mo.position.value = [q]
        return 0

# ===================== Scene =====================
def createScene(root):
    root.dt = DT
    root.gravity = GRAVITY

    # Load aggregate plugins for broad compatibility
    root.addObject('RequiredPlugin', pluginName=[
        'SofaPython3',
        'SofaBaseMechanics','SofaBaseTopology','SofaBaseLinearSolver',
        'SofaGeneralLoader','SofaGeneralTopology','SofaGeneralMeshCollision',
        'SofaGeneralSimpleFem','SofaConstraint','SofaMiscCollision','SofaBoundaryCondition'
    ])

    root.addObject('DefaultVisualManagerLoop')
    root.addObject('VisualStyle', displayFlags='showVisual showBehavior showCollision')

    # Contact pipeline
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)
    root.addObject('DefaultPipeline', depth=6, verbose=0)
    root.addObject('BruteForceDetection')
    root.addObject('LocalMinDistance', alarmDistance=ALARM_DIST, contactDistance=CONTACT_DIST, angleCone=0.0)
    root.addObject('DefaultContactManager', response='FrictionContact', responseParams=f'mu={FRICTION}')

    # ---------- Soft hemisphere ----------
    hemi = root.addChild('Hemisphere')
    hemi.addObject('EulerImplicitSolver', rayleighStiffness=0.02, rayleighMass=0.0)
    hemi.addObject('SparseLDLSolver')

    hemi.addObject('MeshGmshLoader', name='loader', filename=str(HEMI_MESH))
    hemi.addObject('TetrahedronSetTopologyContainer', src='@loader')
    hemi.addObject('TetrahedronSetTopologyModifier')
    hemi.addObject('TetrahedronSetGeometryAlgorithms')
    hemi.addObject('MechanicalObject', name='dofs', src='@loader')
    hemi.addObject('UniformMass', totalMass=DENSITY)
    hemi.addObject('TetrahedronFEMForceField', name='fem',
                   youngModulus=YOUNG, poissonRatio=POISSON, method='corotational')

    # Fix nodes on flat base (z≈0)
    hemi.addObject('BoxROI', name='baseROI', box=[-1, -1, -1e-6, 1, 1, 1e-6])
    hemi.addObject('FixedConstraint', indices='@baseROI.indices')

    # Soft-body collision model (mapped from volume)
    hemiCol = hemi.addChild('Collision')
    hemiCol.addObject('MechanicalObject', src='@../loader')
    hemiCol.addObject('TetrahedronSetTopologyContainer', src='@../loader')
    hemiCol.addObject('TetrahedronSetGeometryAlgorithms')
    hemiCol.addObject('TetrahedronCollisionModel', moving=True, simulated=True)
    hemiCol.addObject('LineCollisionModel')
    hemiCol.addObject('PointCollisionModel')
    hemiCol.addObject('IdentityMapping')

    # Optional visual
    vis = hemi.addChild('Visual')
    vis.addObject('OglModel', src='@../loader', color=[0.6, 0.8, 1.0, 1.0])
    vis.addObject('IdentityMapping')

    # ---------- Rigid indenter ----------
    ind = root.addChild('Indenter')
    ind.addObject('EulerImplicitSolver')
    ind.addObject('SparseLDLSolver')
    # [x y z qx qy qz qw] — start above the hemisphere
    ind.addObject('MechanicalObject', name='dofs', template='Rigid3d',
                  position=[0.0, 0.0, 0.07, 0.0, 0.0, 0.0, 1.0])
    ind.addObject('UniformMass', totalMass=1.0)

    indVis = ind.addChild('Visual')
    indVis.addObject('MeshSTLLoader', name='vloader', filename=str(IND_STL))
    indVis.addObject('OglModel', src='@vloader', color=[0.9, 0.9, 0.9, 1.0])
    indVis.addObject('RigidMapping')

    indCol = ind.addChild('Collision')
    indCol.addObject('MeshSTLLoader', name='cloader', filename=str(IND_STL))
    indCol.addObject('TriangleSetTopologyContainer', src='@cloader')
    indCol.addObject('MechanicalObject', src='@cloader')
    indCol.addObject('TriangleCollisionModel', moving=True, simulated=False)
    indCol.addObject('LineCollisionModel')
    indCol.addObject('PointCollisionModel')
    indCol.addObject('RigidMapping')

    # Monitor reaction forces and pose (writes indentation_force.csv)
    mon = ind.addObject('Monitor', name='forceMonitor', template='Rigid3d', indices=[0],
                        showForces=True, exportForces=True, exportPositions=True,
                        fileName='indentation_force.csv')

    # Controller that drives the indenter
    ctrl = IndentationController(node=ind, rigidMO=ind.getObject('dofs'), monitor=mon)
    ind.addObject(ctrl)

    # Optional camera
    root.addObject('InteractiveCamera', position=[0.2, 0.2, 0.15], lookAt=[0, 0, 0])

    return root
