import torch
from diff_renderer.normal_nds.nds.core import Mesh
from diff_renderer.normal_nds.nds.core.mesh_smpl import SMPLMesh

def laplacian_loss(mesh: Mesh, mask=None):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (Mesh): Mesh used to build the differential coordinates.
    """
    L = mesh.laplacian
    V = mesh.vertices
    
    loss = L.mm(V)
    loss = loss.norm(dim=1)**2
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


def laplacian_loss_canonical(mesh: SMPLMesh):
    """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

    Args:
        mesh (SMPLMesh): Mesh used to build the differential coordinates.
    """
    L = mesh.laplacian
    V = mesh.v_posed

    loss = L.mm(V)
    loss = loss.norm(dim=1) ** 2

    return loss.mean()
